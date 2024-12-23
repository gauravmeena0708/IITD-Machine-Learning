"""
outputs
Epoch [1/20], Train Loss: 0.5537, Test Loss: 0.3792, Test Accuracy: 86.45%
Epoch [2/20], Train Loss: 0.3530, Test Loss: 0.3194, Test Accuracy: 88.35%
Epoch [3/20], Train Loss: 0.3043, Test Loss: 0.2910, Test Accuracy: 89.28%
Epoch [4/20], Train Loss: 0.2767, Test Loss: 0.2800, Test Accuracy: 89.64%
Epoch [5/20], Train Loss: 0.2535, Test Loss: 0.2762, Test Accuracy: 89.95%
Epoch [6/20], Train Loss: 0.2378, Test Loss: 0.2621, Test Accuracy: 90.34%
Epoch [7/20], Train Loss: 0.2206, Test Loss: 0.2467, Test Accuracy: 91.06%
Epoch [8/20], Train Loss: 0.2051, Test Loss: 0.2421, Test Accuracy: 90.95%
Epoch [9/20], Train Loss: 0.1920, Test Loss: 0.2410, Test Accuracy: 91.34%
Epoch [10/20], Train Loss: 0.1793, Test Loss: 0.2432, Test Accuracy: 91.53%
Epoch [11/20], Train Loss: 0.1683, Test Loss: 0.2428, Test Accuracy: 91.61%
Epoch [12/20], Train Loss: 0.1575, Test Loss: 0.2405, Test Accuracy: 91.56%
Epoch [13/20], Train Loss: 0.1467, Test Loss: 0.2400, Test Accuracy: 91.73%
Epoch [14/20], Train Loss: 0.1378, Test Loss: 0.2370, Test Accuracy: 92.02%
Epoch [15/20], Train Loss: 0.1306, Test Loss: 0.2486, Test Accuracy: 91.72%
Epoch [16/20], Train Loss: 0.1208, Test Loss: 0.2601, Test Accuracy: 91.70%
Epoch [17/20], Train Loss: 0.1146, Test Loss: 0.2560, Test Accuracy: 92.18%
Epoch [18/20], Train Loss: 0.1083, Test Loss: 0.2646, Test Accuracy: 92.23%
Epoch [19/20], Train Loss: 0.1002, Test Loss: 0.2687, Test Accuracy: 92.09%
Epoch [20/20], Train Loss: 0.0949, Test Loss: 0.2682, Test Accuracy: 91.87%
Training complete.

"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# Hyperparameters
batch_size = 64
learning_rate = 0.0005  # Lowered learning rate
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations with normalization only
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST dataset
train_dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a simplified CNN model
class SimplifiedFashionCNN(nn.Module):
    def __init__(self):
        super(SimplifiedFashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimplifiedFashionCNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return test_loss / len(loader), accuracy

# Training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

print("Training complete.")
