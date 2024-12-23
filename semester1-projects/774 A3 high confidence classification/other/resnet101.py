import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 251
CONFIDENCE_THRESHOLD = 0.99  # Confidence threshold for high-confidence predictions
PATIENCE = 50  # Early stopping patience

# Data augmentation and normalization for training and testing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define ResNet block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define ResNet architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet101():
    return ResNet(BasicBlock, [3, 4, 23, 3])

# Test function to evaluate the model
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    high_confidence_correct = 0
    high_confidence_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # High confidence predictions
            high_confidence_mask = confidence > CONFIDENCE_THRESHOLD
            high_confidence_total += high_confidence_mask.sum().item()
            high_confidence_correct += (predicted[high_confidence_mask] == targets[high_confidence_mask]).sum().item()

    test_accuracy = 100.0 * correct / total
    if high_confidence_total > 0:
        high_confidence_accuracy = 100.0 * high_confidence_correct / high_confidence_total
    else:
        high_confidence_accuracy = 0.0

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"High-confidence Predictions: {high_confidence_total}, Percentage Correct High-confidence Predictions: {high_confidence_accuracy:.2f}%")

    return test_accuracy

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_accuracy = 0

    def __call__(self, accuracy, model):
        if self.best_score is None:
            self.best_score = accuracy
            self.save_checkpoint(model)
        elif accuracy <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = accuracy
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves the model when validation accuracy improves."""
        self.best_accuracy = self.best_score
        torch.save(model.state_dict(), 'sam4_1910_99.pth')
        print(f"Model saved with accuracy: {self.best_accuracy:.2f}%")

# Training function with early stopping
def train_with_early_stopping(model, train_loader, test_loader, optimizer, criterion, epochs, device, patience):
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Check validation accuracy after each epoch
        test_accuracy = test(model, test_loader, device)

        # Early stopping check
        early_stopping(test_accuracy, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

# Custom loss function with high-confidence penalty
class ConfidencePenaltyLoss(nn.Module):
    def __init__(self, confidence_threshold=0.99, penalty_scale=5.0):
        super(ConfidencePenaltyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.confidence_threshold = confidence_threshold
        self.penalty_scale = penalty_scale

    def forward(self, outputs, targets):
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
        
        # Standard cross-entropy loss
        loss = self.cross_entropy_loss(outputs, targets)
        
        # Penalize wrong predictions with high confidence
        incorrect_high_confidence_mask = (predicted != targets) & (confidence > self.confidence_threshold)
        high_confidence_penalty = confidence[incorrect_high_confidence_mask] * self.penalty_scale
        penalty = high_confidence_penalty.sum()
        
        total_loss = loss + penalty
        return total_loss

# Label Smoothing Loss implementation
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        """
        Constructor for the label smoothing loss.
        :param smoothing: Label smoothing factor (between 0 and 1).
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, outputs, targets):
        """
        Forward pass for label smoothing loss.
        :param outputs: Predictions from the model (logits).
        :param targets: Ground truth labels.
        """
        log_probs = F.log_softmax(outputs, dim=-1)
        # Create a smoothed target distribution
        with torch.no_grad():
            num_classes = outputs.size(-1)
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        return loss

# Custom loss that focuses on high-confidence correct predictions
class HighConfidenceCorrectLoss(nn.Module):
    def __init__(self, confidence_threshold=0.99, penalty_scale=1.5, reward_scale=2.0):
        """
        Custom loss to focus on high-confidence correct predictions.
        :param confidence_threshold: Confidence level above which correct predictions are rewarded.
        :param penalty_scale: Scale for penalizing high-confidence incorrect predictions.
        :param reward_scale: Scale for rewarding high-confidence correct predictions.
        """
        super(HighConfidenceCorrectLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.confidence_threshold = confidence_threshold
        self.penalty_scale = penalty_scale
        self.reward_scale = reward_scale

    def forward(self, outputs, targets):
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

        # Standard cross-entropy loss
        base_loss = self.cross_entropy_loss(outputs, targets)

        # High-confidence correct predictions
        correct_high_confidence_mask = (predicted == targets) & (confidence > self.confidence_threshold)
        incorrect_high_confidence_mask = (predicted != targets) & (confidence > self.confidence_threshold)

        # Reward correct high-confidence predictions
        reward = -confidence[correct_high_confidence_mask].sum() * self.reward_scale

        # Penalize incorrect high-confidence predictions
        penalty = confidence[incorrect_high_confidence_mask].sum() * self.penalty_scale

        total_loss = base_loss + penalty + reward
        return total_loss

# Training function with focus on high-confidence correct predictions
def train_with_high_confidence_focus(model, train_loader, test_loader, optimizer, criterion, epochs, device, patience):
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        high_confidence_correct = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Count high-confidence correct predictions
            probabilities = F.softmax(outputs, dim=1)
            confidence, _ = probabilities.max(1)
            high_confidence_correct += (confidence > CONFIDENCE_THRESHOLD).sum().item()

        train_accuracy = 100.0 * correct / total
        high_confidence_ratio = 100.0 * high_confidence_correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, High-Confidence Correct: {high_confidence_ratio:.2f}%")

        # Check validation accuracy after each epoch
        test_accuracy = test(model, test_loader, device)
        scheduler.step(test_accuracy)
        # Early stopping check
        early_stopping(test_accuracy, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

# Custom loss function that rewards high-confidence correct predictions
class HighConfidenceRewardLoss(nn.Module):
    def __init__(self, reward_scale=1.0):
        super(HighConfidenceRewardLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.reward_scale = reward_scale

    def forward(self, outputs, targets):
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
        
        # Standard cross-entropy loss
        loss = self.cross_entropy_loss(outputs, targets)
        
        # Reward high-confidence correct predictions
        correct_mask = predicted.eq(targets)
        high_confidence_correct = confidence[correct_mask]
        reward = (high_confidence_correct * self.reward_scale).sum()
        
        # Total loss is cross-entropy loss minus the reward
        total_loss = loss - reward / outputs.size(0)  # Normalize by batch size
        return total_loss

# Initialize ResNet-101 and optimizer
model = resnet101()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
criterion = HighConfidenceRewardLoss(reward_scale=1.0)

# Train the model with early stopping
train_with_early_stopping(model, train_loader, test_loader, optimizer, criterion, EPOCHS, device, PATIENCE)



