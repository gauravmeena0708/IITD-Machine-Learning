import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pickle
from trainloader import CustomImageDataset

# Set the seed for reproducibility
torch.manual_seed(0)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 25, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 12 * 25)
        x = self.fc1(x)
        return x

# Define training function
def train_model(train_loader, model, criterion, optimizer, num_epochs, save_path):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.float()
            labels = labels.float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
        torch.save(model.state_dict(), save_path)

# Define testing function
def test_model(test_loader, model, load_path, save_predictions_path):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.float()
            outputs = model(images)
            preds = torch.sigmoid(outputs).round().numpy()
            predictions.extend(preds.flatten())
    
    # Save predictions as a pickle file
    with open(save_predictions_path, 'wb') as f:
        pickle.dump(np.array(predictions), f)

# Example usage
if __name__ == "__main__":
    # Initialize dataset and dataloaders
    # Assuming CustomImageDataset is defined elsewhere and imported
    train_dataset = CustomImageDataset(root_dir='./binary_dataset/',csv = "./binary_dataset/public_train.csv", transform=transforms)
    test_dataset = CustomImageDataset(root_dir='./binary_dataset/',csv = "./binary_dataset/public_test.csv", transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Instantiate the model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(train_loader, model, criterion, optimizer, num_epochs=8, save_path="part_a_binary_model.pth")

    # Test the model
    test_model(test_loader, model, load_path="part_a_binary_model.pth", save_predictions_path="predictions.pkl")
