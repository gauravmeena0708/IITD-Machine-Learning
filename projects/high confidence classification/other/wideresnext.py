import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pickle
import pandas as pd
import os
import torchvision.transforms as transforms

# Early stopping parameters
early_stopping_patience = 20  # Number of epochs with no improvement to stop training
best_test_accuracy = 0.0  # Best test accuracy seen so far
epochs_no_improvement = 0  # Counter for how many epochs with no improvement

# Data Augmentation for Training Set
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),             # Random cropping with padding
    transforms.RandomHorizontalFlip(),                # Random horizontal flip
    transforms.RandomRotation(15),                    # Randomly rotate by 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random jitter in brightness, contrast, etc.
    transforms.ToTensor(),                            # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

# Data Transformations for the Test Set (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),                            # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

from PIL import Image

class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform  # Add transform argument

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        
        # Convert the tensor to a PIL image before applying transforms
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Convert torch tensor to numpy array (H, W, C)
            image = Image.fromarray((image * 255).astype('uint8'))  # Convert numpy array to PIL Image
        
        if self.transform:
            image = self.transform(image)  # Apply the transform
        
        return image, label



# Load train and test datasets with the new augmentations for the training set
train_dataset = CIFAR100Dataset('train.pkl', transform=transform_train)
test_dataset = CIFAR100Dataset('test.pkl', transform=transform_test)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# Set paths and other parameters
PENALTY_WEIGHT = 5 # Weight for penalizing incorrect predictions after 50% accuracy
SAVE_PATH = './saved_models/'  # Directory to save model checkpoints
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Temperature Scaling class
class TemperatureScaling(nn.Module):
    def __init__(self, init_temp=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        return logits / self.temperature

# Focal Loss for handling imbalanced datasets
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        BCE_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Get the probability
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

# Custom Loss Function with Focal Loss and Penalty for Wrong Predictions after 50% Accuracy
def custom_loss_function(outputs, targets, current_accuracy):
    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs, dim=1)
    
    # Get the max probability (confidence) and corresponding predicted class
    confidences, predicted_classes = torch.max(probabilities, dim=1)
    
    # Calculate the Focal Loss for class imbalance
    focal_loss = FocalLoss()(outputs, targets)
    
    # Heavily penalize wrong argmax predictions if training accuracy > 50%
    wrong_predictions = (predicted_classes != targets).float()
    if current_accuracy > 0.5:
        wrong_prediction_penalty = PENALTY_WEIGHT * wrong_predictions.sum()
    else:
        wrong_prediction_penalty = 0

    # Calculate the total loss
    total_loss = focal_loss + wrong_prediction_penalty
    return total_loss

# WideResNeXt Block
class WideResNeXtBlock(nn.Module):
    expansion = 2  # Expansion factor for WideResNeXt

    def __init__(self, in_planes, planes, stride=1, cardinality=32, widen_factor=2):
        super(WideResNeXtBlock, self).__init__()
        D = cardinality * widen_factor
        self.conv1 = nn.Conv2d(in_planes, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, planes * WideResNeXtBlock.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * WideResNeXtBlock.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * WideResNeXtBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * WideResNeXtBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * WideResNeXtBlock.expansion)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# WideResNeXt Model with Temperature Scaling
class WideResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality=32, widen_factor=2, num_classes=100):
        super(WideResNeXt, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, cardinality=cardinality, widen_factor=widen_factor)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, cardinality=cardinality, widen_factor=widen_factor)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, cardinality=cardinality, widen_factor=widen_factor)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, cardinality=cardinality, widen_factor=widen_factor)

        self.dropout = nn.Dropout(p=0.5)  # Add Dropout layer with 0.5 probability
        self.linear = nn.Linear(512 * WideResNeXtBlock.expansion, num_classes)
        self.temperature_scaling = TemperatureScaling()  # Temperature scaling layer

    def _make_layer(self, block, planes, num_blocks, stride, cardinality, widen_factor):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, cardinality, widen_factor))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)  # Apply Dropout before the final linear layer
        out = self.linear(out)
        out = self.temperature_scaling(out)  # Apply temperature scaling before softmax
        return out

def train_with_penalty(epoch):
    model.train()  # Set the model to training mode
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()  # Zero the gradients for the optimizer
        
        outputs = model(inputs)  # Forward pass
        
        # Calculate the overall training accuracy before updating weights
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_classes = torch.max(probabilities, dim=1)
        
        correct_predictions = predicted_classes.eq(targets).sum().item()
        total += targets.size(0)
        current_accuracy = correct_predictions / total
        
        # Calculate the custom loss with penalties if training accuracy > 50%
        loss = custom_loss_function(outputs, targets, current_accuracy)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        train_loss += loss.item()
        correct += correct_predictions

        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {train_loss / (batch_idx + 1):.3f}, Acc: {100.*correct/total:.3f}%')

    # At the end of the epoch, print the final training accuracy
    print(f'Epoch {epoch} Training Loss: {train_loss / len(trainloader):.3f}, Accuracy: {100.*correct/total:.3f}%')

import csv

# Function to calculate accuracy based on True_label data in test_info.csv and save predictions and probabilities
def test_accuracy(epoch, test_info_path):
    global best_test_accuracy, epochs_no_improvement
    model.eval()
    correct_all = 0
    total_all = 0

    # Load True_label from test_info.csv
    test_info = pd.read_csv(test_info_path)
    true_labels = test_info['True_label'].values
    ids = test_info['ID'].values  # Assuming the test_info.csv also contains 'ID' column

    predictions = []  # Store predictions
    probabilities_list = []  # Store probabilities for each sample

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)  # Get probabilities for each class
            _, predicted_classes = torch.max(probabilities, dim=1)  # Get predicted class

            # Store predictions and probabilities
            predictions.extend(predicted_classes.cpu().numpy())
            probabilities_list.extend(probabilities.cpu().numpy())  # Save softmax probabilities for each sample

    # Calculate accuracy based on True_label
    correct_all = (predictions == true_labels).sum()
    total_all = len(true_labels)

    # Print accuracy
    test_accuracy = 100. * correct_all / total_all
    print(f"Epoch {epoch}, Test Accuracy: {test_accuracy:.2f}%")

    # Save predictions and probabilities to CSV after epoch 70
    if epoch >= 70:
        save_predictions_and_probabilities_to_csv(ids, predictions, probabilities_list, epoch)

    # Check for improvement and early stopping condition
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        epochs_no_improvement = 0  # Reset the counter when improvement is seen
        # Optionally save the model
        save_model_checkpoint(epoch)
    else:
        epochs_no_improvement += 1  # No improvement
        print(f'No improvement for {epochs_no_improvement} epoch(s)')
    
    # Stop if no improvement for 10 epochs
    if epochs_no_improvement >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch} due to no improvement for {early_stopping_patience} epochs.")
        return True  # Signal to stop training
    return False

def save_predictions_and_probabilities_to_csv(ids, predictions, probabilities, epoch):
    """
    Save the predictions and probabilities to a CSV file.
    The CSV will contain the following columns: ID, Predicted_label, and probabilities for each class (Prob_Class0, Prob_Class1, ..., Prob_Class99).
    """
    probabilities_df = pd.DataFrame(probabilities, columns=[f'Prob_Class{i}' for i in range(100)])
    predictions_df = pd.DataFrame({'ID': ids, 'Predicted_label': predictions})
    
    # Combine the predictions and probabilities into a single DataFrame
    df = pd.concat([predictions_df, probabilities_df], axis=1)
    
    # Save to CSV with the epoch number in the filename
    csv_filename = f'predictions_and_probabilities_epoch_{epoch}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Predictions and probabilities saved to {csv_filename}")


def save_model_checkpoint(epoch):
    save_path = os.path.join(SAVE_PATH, f'best_model_{str(epoch)}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"New best model saved with accuracy: {best_test_accuracy:.2f}%")

# Model, loss, optimizer, and scheduler
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WideResNeXt(WideResNeXtBlock, [3, 4, 6, 3], cardinality=32, widen_factor=2).to(device)

# Example optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer)


# Training loop with scheduler switching
for epoch in range(0, 250):
    train_with_penalty(epoch)  # Perform training for this epoch
    stop_training = test_accuracy(epoch, 'test_info.csv')  # Evaluate test accuracy


    scheduler.step()

    if stop_training:
        break  # Stop the training loop if early stopping is triggered


