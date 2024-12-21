import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import time
import json
from pyswarm import pso

# Set the total countdown time (in seconds, 10 minutes = 600 seconds)
total_time = 10 * 60

def time_remaining(start_time, total_time):
    elapsed_time = time.time() - start_time
    remaining_time = total_time - elapsed_time
    if remaining_time < 0:
        remaining_time = 0
    minutes, seconds = divmod(int(remaining_time), 60)
    return f'{minutes} minutes {seconds} seconds'

np.random.seed(0)

# Custom Dataset Class
class CustomImageDataset:
    def __init__(self, root_dir, csv, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("L")  # Convert image to greyscale
        label = row["class"]

        if self.transform:
            image = self.transform(image)

        return np.array(image), label

# Transformations using NumPy
def resize(image, size):
    return np.array(image.resize(size))

def to_tensor(image):
    return image.astype(np.float32) / 255.0

def numpy_transform(image, size=(25, 25)):
    image = resize(image, size)
    image = to_tensor(image)
    image = image.flatten()
    return image

# DataLoader Class
class DataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        self.start_idx = 0
        return self

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __next__(self):
        if self.start_idx >= len(self.dataset):
            raise StopIteration

        end_idx = min(self.start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.start_idx:end_idx]
        images = []
        labels = []

        for idx in batch_indices:
            image, label = self.dataset[idx]
            images.append(image)
            labels.append(label)

        self.start_idx = end_idx

        batch_images = np.stack(images, axis=0)
        batch_labels = np.array(labels)

        return batch_images, batch_labels

def initialize_weights(sizes):
    weights = {}
    biases = {}

    weights['fc1'] = np.random.randn(sizes['input'], sizes['hidden'][0]) * np.sqrt(2 / sizes['input'])
    biases['fc1'] = np.zeros((1, sizes['hidden'][0]))
    hidden_layer_sizes = sizes['hidden']
    for i in range(1, len(hidden_layer_sizes)):
        weights[f'fc{i+1}'] = np.random.randn(hidden_layer_sizes[i-1], hidden_layer_sizes[i]) * np.sqrt(2 / hidden_layer_sizes[i-1])
        biases[f'fc{i+1}'] = np.zeros((1, hidden_layer_sizes[i]))

    weights[f'fc{len(hidden_layer_sizes)+1}'] = np.random.randn(hidden_layer_sizes[-1], sizes['output']) * np.sqrt(2 / hidden_layer_sizes[-1])
    biases[f'fc{len(hidden_layer_sizes)+1}'] = np.zeros((1, sizes['output']))

    return weights, biases

# Sigmoid activation function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def dropout(A, dropout_rate):
    mask = (np.random.rand(*A.shape) < (1 - dropout_rate)) / (1 - dropout_rate)
    A *= mask
    return A

def forward_pass(X, weights, bias, params, training=False):
    layers = len(params['hidden']) + 1  
    activations = [X]
    Z_linear = []

    for i in range(1, layers):
        Z_current = np.dot(activations[-1], weights[f'fc{i}']) + bias[f'fc{i}']
        A_current = sigmoid(Z_current)
        
        if training:
            A_current = dropout(A_current, dropout_rate=params["dropout"])

        Z_linear.append(Z_current)
        activations.append(A_current)

    Z_out = np.dot(activations[-1], weights[f'fc{layers}']) + bias[f'fc{layers}']
    A_out = softmax(Z_out)

    Z_linear.append(Z_out)
    activations.append(A_out)

    return Z_linear, activations

def compute_loss(ground_truth, predictions, num_classes=8):
    ground_truth_one_hot = np.eye(num_classes)[ground_truth]
    predictions_one_hot = np.eye(num_classes)[predictions]
    
    epsilon = 1e-12
    predictions_one_hot = np.clip(predictions_one_hot, epsilon, 1. - epsilon)
    cross_entropy = -np.sum(ground_truth_one_hot * np.log(predictions_one_hot)) / len(predictions)
    return cross_entropy

# Adam optimizer implementation
def adam_optimizer(weights, bias, dW, db, vW, vb, sW, sb, t, beta1, beta2, epsilon, learning_rate):
    vW = {key: beta1 * vW[key] + (1 - beta1) * dW[key] for key in dW}
    vb = {key: beta1 * vb[key] + (1 - beta1) * db[key] for key in db}
    
    sW = {key: beta2 * sW[key] + (1 - beta2) * (dW[key] ** 2) for key in dW}
    sb = {key: beta2 * sb[key] + (1 - beta2) * (db[key] ** 2) for key in db}

    vW_corrected = {key: vW[key] / (1 - beta1 ** t) for key in vW}
    vb_corrected = {key: vb[key] / (1 - beta1 ** t) for key in vb}
    sW_corrected = {key: sW[key] / (1 - beta2 ** t) for key in sW}
    sb_corrected = {key: sb[key] / (1 - beta2 ** t) for key in sb}

    updated_weights = {key: weights[key] - learning_rate * vW_corrected[key] / (np.sqrt(sW_corrected[key]) + epsilon) for key in weights}
    updated_biases = {key: bias[key] - learning_rate * vb_corrected[key] / (np.sqrt(sb_corrected[key]) + epsilon) for key in bias}

    return updated_weights, updated_biases, vW, vb, sW, sb

def backward_pass(X, Y, Z, activations, weights, bias, params, vW, vb, sW, sb, t):
    m = X.shape[0]
    layers = len(params['hidden']) + 1

    dW = {}
    db = {}

    dZ_out = activations[-1] - Y
    dW[f'fc{layers}'] = np.dot(activations[-2].T, dZ_out) / m
    db[f'fc{layers}'] = np.sum(dZ_out, axis=0, keepdims=True) / m
    dA_prev = np.dot(dZ_out, weights[f'fc{layers}'].T)

    for i in range(layers - 1, 0, -1):
        dZ_current = dA_prev * sigmoid_derivative(activations[i])
        dW[f'fc{i}'] = np.dot(activations[i - 1].T, dZ_current) / m
        db[f'fc{i}'] = np.sum(dZ_current, axis=0, keepdims=True) / m
        if i > 1:
            dA_prev = np.dot(dZ_current, weights[f'fc{i}'].T)

    weights, bias, vW, vb, sW, sb = adam_optimizer(weights, bias, dW, db, vW, vb, sW, sb, t, params['beta1'], 0.999, 1e-8, params['rate'])

    return weights, bias, vW, vb, sW, sb

# Training function
def train_neural_network(dataloader, params, dataloader_test):
    print("start training..")
    weights, bias = initialize_weights(params)
    vW = {key: np.zeros_like(val) for key, val in weights.items()}
    vb = {key: np.zeros_like(val) for key, val in bias.items()}
    sW = {key: np.zeros_like(val) for key, val in weights.items()}
    sb = {key: np.zeros_like(val) for key, val in bias.items()}
    
    epochs = params['epochs']
    t = 1
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, Y_batch in dataloader:
            Y_batch_one_hot = np.eye(params['output'])[Y_batch]  
            Zs, As = forward_pass(X_batch, weights, bias, params, True)
            weights, bias, vW, vb, sW, sb = backward_pass(X_batch, Y_batch_one_hot, Zs, As, weights, bias, params, vW, vb, sW, sb, t)
            t += 1
            epoch_loss += compute_loss(Y_batch, np.argmax(As[-1], axis=1))
        
        epoch_loss /= len(dataloader)
        remaining = time_remaining(start_time, total_time)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f} Time remaining: {remaining}")
        if time.time() - start_time >= total_time:
            print("Time's up!")
            break
        time.sleep(3)
    
    return weights, bias

def test_neural_network(dataloader, weights, bias, params):
    test_loss = 0
    predictions = []
    for X_batch, Y_batch in dataloader:
        Y_batch_one_hot = np.eye(params['output'])[Y_batch]  
        Zs, As = forward_pass(X_batch, weights, bias, params)
        test_loss += compute_loss(Y_batch, np.argmax(As[-1], axis=1))
        predicted_classes = np.argmax(As[-1], axis=1)
        predictions.extend(predicted_classes)
    test_loss /= len(dataloader)
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss, predictions


def save_weights(weights, bias, save_path):
    biases = {}
    for key in bias:
        biases[key] = bias[key].flatten()
    model = {'weights': weights, 'bias': biases}
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

import random

# Function to evaluate the neural network for PSO
def evaluate_hyperparameters(params):
    rate, beta1, batch_size = params

    # Set the hyperparameters
    hyper_params = {
        'input': 625,
        'hidden': [512, 256, 128, 32],
        'output': 8,
        'rate': round(rate, 4),
        'epochs': 4,
        'dropout': 0.0,
        'beta1': round(beta1, 2),
        'batch_size': int(batch_size)
    }

    # Update dataloader with new batch size
    dataloader_train = DataLoader(dataset_train, int(batch_size))
    dataloader_test = DataLoader(dataset_test, 256)  # Keeping test batch size fixed for simplicity

    # Train the model and evaluate test loss
    weights, bias = train_neural_network(dataloader_train, hyper_params, dataloader_test)
    test_loss, prediction = test_neural_network(dataloader_train, weights, bias, hyper_params)

    # Log trial results to JSON
    log_entry = {
        'learning_rate': round(rate, 4),
        'beta1': round(beta1, 2),
        'dropout': 0.0,
        'batch_size': int(batch_size),
        'loss': test_loss
    }

    # Append the new log entry to the log.json file
    if os.path.exists("log.json"):
        with open("log.json", "r") as log_file:
            logs = json.load(log_file)
    else:
        logs = []

    logs.append(log_entry)

    with open("log.json", "w") as log_file:
        json.dump(logs, log_file, indent=4)

    return test_loss

# Main code
if __name__ == '__main__':
    start_time = time.time()
    root_dir = ".//dataset_for_A2//multi_dataset//"
    csv_train = os.path.join(root_dir, "train.csv")
    csv_test = os.path.join(root_dir, "val.csv")
    
    dataset_train = CustomImageDataset(root_dir=root_dir, csv=csv_train, transform=numpy_transform)
    dataset_test = CustomImageDataset(root_dir=root_dir, csv=csv_test, transform=numpy_transform)

    # Define bounds for PSO
    lb = [0.0001, 0.70, 16]  # Lower bounds for learning rate, beta1, dropout, batch size
    ub = [0.001, 0.99, 56]   # Upper bounds for learning rate, beta1, dropout, batch size

    # Perform PSO to find the best hyperparameters
    best_params, best_loss = pso(evaluate_hyperparameters, lb, ub, swarmsize=5, maxiter=5)

    print(f"Best Hyperparameters: Learning Rate={best_params[0]}, Beta1={best_params[1]}, Batch Size={int(best_params[3])}")
    print(f"Best Loss: {best_loss:.4f}")
