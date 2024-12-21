import pickle
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import random
from collections import defaultdict
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR

# Set fixed seed values for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-100 Dataset class
class CIFAR100Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f'Unsupported image type: {type(image)}')

        if self.transform:
            image = self.transform(image)

        return image, label

# Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Added rotation for more diversity
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # Normalization for CIFAR-100
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),  # Cutout
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Custom sampler to ensure 10 images per class in each batch
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, num_classes, images_per_class):
        self.dataset = dataset
        self.num_classes = num_classes
        self.images_per_class = images_per_class
        self.batch_size = num_classes * images_per_class
        self.class_to_indices = defaultdict(list)

        # Group all indices by their label (class)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.class_to_indices[label].append(idx)

        self.num_batches = len(dataset) // self.batch_size

    def __iter__(self):
        # Randomly shuffle class indices and prepare batches
        class_indices = {label: random.sample(indices, len(indices)) for label, indices in self.class_to_indices.items()}
        batches = []
        
        for _ in range(self.num_batches):
            batch = []
            for label in range(self.num_classes):
                if len(class_indices[label]) >= self.images_per_class:
                    # Take exactly 10 images per class
                    batch.extend(class_indices[label][:self.images_per_class])
                    # Remove those images from the list
                    class_indices[label] = class_indices[label][self.images_per_class:]
                else:
                    continue
            random.shuffle(batch)
            batches.append(batch)
        
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.num_batches

# Mixup function
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Label Smoothing Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        one_hot = torch.zeros_like(log_prob).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / log_prob.size(1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss

# Load datasets
trainset_full = CIFAR100Dataset('train.pkl', transform=train_transform)
testset = CIFAR100Dataset('test.pkl', transform=test_transform)

# Split trainset into training and validation sets
train_size = int(0.9 * len(trainset_full))
val_size = len(trainset_full) - train_size
trainset, valset = random_split(trainset_full, [train_size, val_size])

# Number of classes in CIFAR-100
num_classes = 100
images_per_class = 10  # Now you want 10 images per class in each batch
batch_size = num_classes * images_per_class  # This will be 1000 in this case

# Use the BalancedBatchSampler for the training set
trainloader = DataLoader(trainset, batch_sampler=BalancedBatchSampler(trainset, num_classes, images_per_class))
valloader = DataLoader(valset, batch_size=64, shuffle=False)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Define WideResNet model
def get_model():
    from torchvision.models import wide_resnet101_2
    model = wide_resnet101_2(num_classes=100)
    return model

# Define training function
def train_model(model, criterion, optimizer, scheduler, swa_model=None, swa_scheduler=None, num_epochs=20):
    scaler = GradScaler()
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1.0)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Scheduler step
        scheduler.step()

        # Apply SWA
        if swa_model and swa_scheduler:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Validation accuracy
        val_acc = validate_model(model)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Val Acc: {val_acc:.4f}')

    return best_acc

# Define validation function
def validate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Test function for high confidence predictions
def test_high_confidence(model):
    model.eval()
    high_conf_count = 0
    total = 0
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)

            high_conf_count += (max_probs > 0.99).sum().item()
            total += inputs.size(0)
    return high_conf_count, total

# Run experiments with different learning rates and optimizers
learning_rates = [0.1, 0.05, 0.01]
optimizers = [optim.AdamW, optim.SGD]  # AdamW for better weight decay handling
results = []

for lr in learning_rates:
    for opt in optimizers:
        print(f"Testing WideResNet with LR={lr} and Optimizer={opt.__name__}")
        model = get_model().to(device)
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # Create optimizer and scheduler
        optimizer = opt(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=5, mode='triangular2')

        # SWA model and scheduler
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)

        # Train the model
        best_val_acc = train_model(model, criterion, optimizer, scheduler, swa_model, swa_scheduler, num_epochs=30)

        # Load best model and test high confidence predictions
        model.load_state_dict(torch.load("best_model.pth"))
        high_conf_count, total = test_high_confidence(model)
        high_conf_percentage = high_conf_count / total * 100

        results.append({
            "lr": lr,
            "optimizer": opt.__name__,
            "val_acc": best_val_acc,
            "high_conf_percentage": high_conf_percentage
        })

        print(f"LR: {lr}, Optimizer: {opt.__name__}, Val Acc: {best_val_acc}, High Confidence: {high_conf_percentage}%\n")

# Display the best result
best_result = max(results, key=lambda x: x['val_acc'])
print("Best Result:")
print(best_result)

