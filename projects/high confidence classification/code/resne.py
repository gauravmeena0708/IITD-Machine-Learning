import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CyclicLR
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torchvision.datasets as datasets
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnext101_32x8d
from torch.utils.data import DataLoader
import pickle
from PIL import Image
import numpy as np

class CIFAR100Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform  # Store the transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        
        # Check if the image is a NumPy array and convert it to a PIL image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply the transform if specified
        if self.transform:
            image = self.transform(image)

        return image, label

# Define your transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),  # This should be skipped if image is already a tensor
])

# Pass the transform to the dataset
trainset = CIFAR100Dataset('train.pkl', transform=train_transform)
testset = CIFAR100Dataset('test.pkl')  # Applying ToTensor for test




startTime = time.time()



# Hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs = 100
confidence_threshold = 0.99  # Confidence threshold (99%)
alpha = 0.1  # Weight for the confidence penalty


# Initialize the DataLoaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define the ResNeXt model
class CustomResNeXt(nn.Module):
    def __init__(self):
        super(CustomResNeXt, self).__init__()
        self.model = resnext101_32x8d(pretrained=False)  # ResNeXt50 (32x4d) architecture
        self.model.fc = nn.Linear(2048, 100)  # Modify the last fully connected layer for 100 classes (CIFAR-100)

    def forward(self, x):
        return self.model(x)

model = CustomResNeXt().cuda()

# Custom loss function with confidence penalty
def custom_loss_function(outputs, labels, alpha):
    ce_loss = nn.CrossEntropyLoss()(outputs, labels)  # Standard cross-entropy loss
    softmax_outputs = torch.softmax(outputs, dim=1)
    max_probabilities, _ = torch.max(softmax_outputs, dim=1)
    
    # Confidence penalty term: reward higher confidence
    confidence_penalty = -torch.log(max_probabilities).mean()
    
    # Combined loss: cross-entropy loss + alpha * confidence penalty
    total_loss = ce_loss + alpha * confidence_penalty
    return total_loss

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    high_confidence_count = 0  # Count of predictions with confidence > 99%

    for inputs, labels in trainloader:
        inputs, labels = inputs.cuda(), labels.cuda()

        # Forward pass
        outputs = model(inputs)
        loss = custom_loss_function(outputs, labels, alpha)  # Use the custom loss function

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Calculate softmax probabilities
        probabilities = torch.softmax(outputs, dim=1)
        max_probabilities, _ = torch.max(probabilities, dim=1)

        # Count how many have max probability greater than 99%
        high_confidence_count += (max_probabilities > confidence_threshold).sum().item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}, Accuracy: {100.*correct/total}%')
    print(f'Number of high-confidence (>99%) predictions: {high_confidence_count} out of {total}')

# Save the trained model
torch.save(model.state_dict(), 'resnext_model.pth')
