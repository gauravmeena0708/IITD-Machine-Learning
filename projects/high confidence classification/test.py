import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse
import numpy as np
import random
import scipy.special

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

class CIFAR100Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        
        
        if self.transform:
            image = self.transform(image)

        return image, label



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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('modelPath', type=str, help="Model Path")
    parser.add_argument('testPath', type=str, help="Test file Path")
    parser.add_argument('alpha', type=str, help='alpha')
    parser.add_argument('gamma', type=str, help='gamma')

    args = parser.parse_args()

    transform_test = transforms.Compose([
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    testset = CIFAR100Dataset(args.testPath, transform=transform_test)

    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model = cnnModel(depth=28, num_classes=100, widen_factor=10, drop_rate=0.3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.load_state_dict(torch.load(args.modelPath, map_location=torch.device('cpu')))

    model.eval()
    predictions = []
    test_ids = []
    final_groups = {0: {'keys': [53, 26, 39, 68, 15, 48, 99, 49, 40, 88, 90, 50, 55, 47, 33, 37], 'max_value': 0.9998}, 1: {'keys': [21, 72, 52, 51, 10, 96, 89, 46, 79, 17, 62, 78, 45, 25, 92, 22], 'max_value': 0.97}, 2: {'keys': [28, 54, 8, 19, 5, 98, 77, 30, 6, 70, 94, 24, 86, 81, 76, 74], 'max_value': 0.95}, 3: {'keys': [35, 58, 2, 57, 60, 0, 20, 9, 71, 11, 16, 12, 61, 73, 63, 14], 'max_value': 0.90}, 4: {'keys': [1, 69, 36, 82, 95, 80, 97, 34, 91, 83, 3, 42, 43, 64, 4, 31], 'max_value': 0.85}, 5: {'keys': [65, 13, 87, 18, 38, 84, 56, 32, 59, 29, 85, 44, 23, 41, 7, 75], 'max_value': 0.80}}

    with torch.no_grad():
        for inputs, uniqueId in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            scaled_logits = outputs / 3.8
            probabilities = scipy.special.softmax(scaled_logits.cpu().numpy(), axis=1)
                

            max_probs = probabilities.max(axis=1)
            predicted_labels = probabilities.argmax(axis=1)

            class_thresholds = {}
            for group in final_groups.values():
                keys = group["keys"]
                threshold = group["max_value"]
                for key in keys:
                    class_thresholds[key] = threshold

            thresholds = np.array([class_thresholds.get(label, 1.0) for label in predicted_labels])

            final_labels = np.where(max_probs < thresholds, -1, predicted_labels)

            predictions.extend(final_labels)
            test_ids.extend(uniqueId.cpu().numpy())



    submission_data = {'ID': test_ids, 'Predicted_label': predictions}

    submission_df = pd.DataFrame(submission_data)

    submission_df.to_csv('submission.csv', index=False)
