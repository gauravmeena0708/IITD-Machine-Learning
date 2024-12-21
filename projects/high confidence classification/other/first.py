import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to load CIFAR-100 file
def load_cifar100(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=100):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Depth should be 6n+4'
        n = (depth - 4) // 6

        k = widen_factor
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(BasicBlock, 16 * k, n, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32 * k, n, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n, stride=2)
        self.linear = nn.Linear(64 * k, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

import pickle

def load_cifar100_data(train_path, test_path):
    # Load the training data
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
    
    # Load the test data
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    
    # Display properties of the training data
    print("Training Data Properties:")
    print(f"Keys: {train_data.keys()}")
    print(f"Number of training samples: {len(train_data[b'data'])}")
    print(f"Shape of each training image (flattened): {train_data[b'data'][0].shape}")  # This will show as (3072,)
    print(f"Shape of each training image (reshaped): {train_data[b'data'][0].reshape(3, 32, 32).shape}")  # This will show as (3, 32, 32)
    print(f"Number of training labels: {len(train_data[b'fine_labels'])}")
    
    # Display properties of the test data
    print("\nTest Data Properties:")
    print(f"Keys: {test_data.keys()}")
    print(f"Number of test samples: {len(test_data[b'data'])}")
    print(f"Shape of each test image (flattened): {test_data[b'data'][0].shape}")  # This will show as (3072,)
    print(f"Shape of each test image (reshaped): {test_data[b'data'][0].reshape(3, 32, 32).shape}")  # This will show as (3, 32, 32)
    #print(f"Number of test IDs: {len(test_data[b'ids'])}")
    
    # Return the data for further use
    train_images = train_data[b'data']
    train_labels = train_data[b'fine_labels']
    test_images = test_data[b'data']
    test_ids = test_data[b'filenames']

    return train_images, train_labels, test_images, test_ids



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(3, 32, 32).astype(np.uint8)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

train_images, train_labels, test_images, test_ids = load_cifar100_data('/home/aizceq/nbs/A3/cifar-100-python/train', '/home/aizceq/nbs/A3/cifar-100-python/test')

train_dataset = CIFAR100Dataset(train_images, train_labels, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

test_dataset = CIFAR100Dataset(test_images, np.zeros(len(test_images)), transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

def train(model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()
    print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader)}, Accuracy: {100.*correct/total}")

def test(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to('cuda')
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())

    return predictions

model = WideResNet(depth=28, widen_factor=10, num_classes=100)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

num_epochs = 200
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, scheduler, epoch)

# Save the model
torch.save(model.state_dict(), "model.pth")

# Inference on the test set
predictions = test(model, test_loader)

# Prepare the submission file
submission = np.array([test_ids, predictions]).T
np.savetxt('submission.csv', submission, fmt='%d,%d', header='ID,Predicted', comments='')
