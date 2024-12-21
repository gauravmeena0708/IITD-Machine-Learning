import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CyclicLR, StepLR
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import argparse
import time
import random
import pandas as pd

startTime = time.time()
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

set_seed(42)

def load_data(file_path):
    # Load data from a pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
    
class CIFAR100Dataset(Dataset):
    def __init__(self, file_path, transform=None, id_to_label=None):
        # Load your data here
        self.data = load_data(file_path)  # Load the data from the pickle file
        self.transform = transform
        self.id_to_label = id_to_label  # Optional; only used for test set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming data[idx] returns (image, uniqueId)
        image, uniqueId = self.data[idx]
        
        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
        
        # Retrieve the label if id_to_label is provided
        label = self.id_to_label.get(uniqueId, -1) if self.id_to_label else None

        if label is not None:
            return image, uniqueId, label  # For test set
        else:
            return image, uniqueId  # For train set or others


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.3):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        out = self.relu(self.bn1(x))
        if not self.equalInOut:
            x = out
        out = self.relu(self.bn2(self.conv1(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.convShortcut is None else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.3):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class cnnModel(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.3):
        super(cnnModel, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, 'Depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlock, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.fc.in_features)
        return self.fc(out)

def check_accuracy(model,testloader):
    # Validation loop for early stopping
    model.eval()
    correct = 0
    total = 1
    with torch.no_grad():
        for inputs, uniqueId,labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    return val_acc

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('trainPath', type=str, help="Train File Path")
    # parser.add_argument('alpha', type=str, help='alpha')
    # parser.add_argument('gamma', type=str, help='gamma')

    # args = parser.parse_args()


    train_transform = transforms.Compose([

        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),  # Convert to uint8
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.Lambda(lambda x: x.to(torch.float32) / 255.0),  # Convert back to float32 in range [0, 1]
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    testinfodf = pd.read_csv('test_info.csv')
    id_to_label = dict(zip(testinfodf['ID'], testinfodf['True_label']))

    trainset = CIFAR100Dataset('train.pkl', transform=train_transform)


    testset = CIFAR100Dataset('test.pkl', transform=transform_test, id_to_label=id_to_label)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model = cnnModel(depth=28, num_classes=100, widen_factor=10, drop_rate=0.3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = 128
    learning_rate = 0.01
    num_epochs = 110

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=20)
    scaler = GradScaler() 

    for epoch in range(num_epochs):
        
                
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
    
        if epoch == 80:
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        val_acc = check_accuracy(model,test_loader)

        endTime = time.time()
        diff = endTime-startTime
        hours, rem = divmod(diff, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(trainloader)}, accuracy: {val_acc:.2f}% , time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)}')
        if epoch+1 >=75:
            name = f"model_{epoch+1}.pth"
            torch.save(model.state_dict(),name) 