import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time, pickle

import argparse

np.random.seed(0)

class CustomImageDataset:
    def __init__(self, root_dir, csv, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("L") #Convert image to greyscale
        label = row["class"]

        if self.transform:
            image = self.transform(image)

        return np.array(image), label

# Transformations using NumPy
def resize(image, size):
    # return np.array(Image.fromarray(image).resize(size))
    return np.array(image.resize(size))

def to_tensor(image):
    return image.astype(np.float32) / 255.0

def numpy_transform(image, size=(25, 25)):
    image = resize(image, size)
    image = to_tensor(image)
    image = image.flatten()
    return image

class DataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        # if self.shuffle:
        #     np.random.shuffle(self.indices)

    def __iter__(self):
        self.start_idx = 0
        return self
    def __len__(self):
        return int(len(self.dataset)/self.batch_size)

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

        # Stack images and labels to create batch tensors
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

def sigmoid(z):
    return 1 / (1 + np.exp(-z + 1e-12))

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
        A_current = sigmoid(Z_current)  # Use sigmoid during both training and testing
        
        if training:
            A_current = dropout(A_current, dropout_rate=params["dropout"])  # Apply dropout only during training

        Z_linear.append(Z_current)
        activations.append(A_current)

    Z_out = np.dot(activations[-1], weights[f'fc{layers}']) + bias[f'fc{layers}']
    A_out = softmax(Z_out)

    Z_linear.append(Z_out)
    activations.append(A_out)

    return Z_linear, activations

def compute_loss(ground_truth_one_hot, predictions_one_hot, num_classes=8):
    epsilon = 1e-12
    predictions_one_hot = np.clip(predictions_one_hot, epsilon, 1. - epsilon)
    cross_entropy = -np.sum(ground_truth_one_hot * np.log(predictions_one_hot)) / len(predictions_one_hot)
    return cross_entropy

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

total_time_2286 = 14 * 60

def time_remaining(start_time, total_time):
    elapsed_time = time.time() - start_time
    remaining_time = total_time - elapsed_time
    if remaining_time < 0:
        remaining_time = 0
    minutes, seconds = divmod(int(remaining_time), 60)
    return f'{minutes} minutes {seconds} seconds'

def train_neural_network(dataloader, params):
    weights, bias = initialize_weights(params)
    vW = {key: np.zeros_like(val) for key, val in weights.items()}
    vb = {key: np.zeros_like(val) for key, val in bias.items()}
    sW = {key: np.zeros_like(val) for key, val in weights.items()}
    sb = {key: np.zeros_like(val) for key, val in bias.items()}
    
    epochs = params['epochs']
    start_time = time.time()
    t = 1
    
    for epoch in range(epochs):
        all_predictions = []
        all_ground_truths = []
        epoch_loss = 0

        for X_batch, Y_batch in dataloader:
            x_std = standardize_test_data(X_batch, params['mean'], params['std'])
            Y_batch_one_hot = np.eye(params['output'])[Y_batch]  
            Zs, As = forward_pass(x_std, weights, bias, params)
            weights, bias, vW, vb, sW, sb = backward_pass(x_std, Y_batch_one_hot, Zs, As, weights, bias, params, vW, vb, sW, sb, t)
            t += 1
            epoch_loss = compute_loss(Y_batch_one_hot, As[-1])

        epoch_loss /= len(dataloader)
        remaining = time_remaining(start_time, total_time_2286)
        if time.time() - start_time >= total_time_2286:
            break
    return weights, bias


def save_weights(weights, bias, save_path):
    biases = {}
    for key in bias:
        biases[key] = bias[key].flatten()
    model = {'weights': weights, 'bias': biases}
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

def load_weights(load_path):
    with open(load_path, 'rb') as f:
        model = pickle.load(f)
    return model['weights'], model['bias']


def standardize_data(data):
    all_images = np.stack([item[0] for item in data], axis=0)
    mean = np.mean(all_images, axis=0)
    std = np.std(all_images, axis=0)
    
    standardized_data = []
    for image_data, label in data:
        standardized_image = (image_data - mean) / (std + 1e-8)  # Adding epsilon to avoid division by zero
        standardized_data.append((standardized_image, label))
    
    return standardized_data, mean, std

    
def standardize_test_data(data, mean, std):
    standardized_data = [(image_data - mean) / (std + 1e-8) for image_data in data]
    return np.array(standardized_data)

def adjust_weights_biases_for_non_standardized_data(weights, biases, mean, std):
    adjusted_weights = {}
    adjusted_biases = {}
    std = std.reshape(-1, 1)
    mean = mean.reshape(-1, 1)

    adjusted_weights['fc1'] = weights['fc1'] / (std + 1e-8)
    adjusted_biases['fc1'] = biases['fc1'] - np.dot(mean.T / (std.T + 1e-8), weights['fc1'])

    for i in range(2, len(weights) + 1):
        adjusted_weights[f'fc{i}'] = weights[f'fc{i}']
        adjusted_biases[f'fc{i}'] = biases[f'fc{i}']

    return adjusted_weights, adjusted_biases





"""
For PSO optimization

def pso(objective_function, lower_bound, upper_bound, n_particles, n_dimensions, max_iter, w, c1, c2):
    particles = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_particles, n_dimensions))
    personal_best_positions = particles
    global_best_position = particles[np.argmin([objective_function(p) for p in particles])]
    velocities = np.zeros((n_particles, n_dimensions))
    for i in range(max_iter):
        r1 = np.random.rand(n_particles, n_dimensions)
        r2 = np.random.rand(n_particles, n_dimensions)
        velocities = w * velocities + c1 * r1 * (personal_best_positions - particles) + c2 * r2 * (global_best_position - particles)
        particles = particles + velocities
        particles = np.clip(particles, lower_bound, upper_bound)
        for j in range(n_particles):
            if objective_function(particles[j]) < objective_function(personal_best_positions[j]):
                personal_best_positions[j] = particles[j]
            if objective_function(personal_best_positions[j]) < objective_function(global_best_position):
                global_best_position = personal_best_positions[j]

    return global_best_position

hyperparameters = pso(evaluate_hyperparameters, lb, ub, n_particles, n_dimensions, max_iter, w, c1, c2)
#print(hyperparameters)
"""

def train_nn(root_dir, weights_path):
    start_time = time.time()
    csv_train = os.path.join(root_dir, "train.csv")
    rate, beta1, batch_size = 0.0019, 0.9, 64
    hyper_params = {
        'input': 625,
        'hidden': [512, 256, 128, 32],
        'output': 8,
        'rate': round(rate, 4),
        'epochs': 150,
        'dropout': 0.0,
        'beta1': round(beta1, 2),
        'batch_size': int(batch_size)
    }

    dataset_train = CustomImageDataset(root_dir=root_dir, csv=csv_train, transform=numpy_transform)
    dataloader_train = DataLoader(dataset_train, hyper_params['batch_size'])

    sample = []
    for X_batch, Y_batch in dataloader_train:
        sample.extend([(x, y) for x, y in zip(X_batch, Y_batch)])
        if len(sample) >= 256:
            break

    sample_standardized, mean, std = standardize_data(sample)
    hyper_params['std'] = std
    hyper_params['mean'] = mean
    
    weights, bias = train_neural_network(dataloader_train, hyper_params)
    adjusted_weights, adjusted_biases = adjust_weights_biases_for_non_standardized_data(weights, bias, hyper_params['mean'], hyper_params['std'])
    save_weights(adjusted_weights, adjusted_biases, weights_path)

if __name__ == '__main__':
    #with assumption
    #python .\part_c.py --dataset_root=.\\dataset_for_A2\\multi_dataset\\ --save_weights_path=weights.pkl
    parser = argparse.ArgumentParser(description="Train a CNN model")

    parser.add_argument('--dataset_root', type=str, required=True, help='dataset root path')
    parser.add_argument('--save_weights_path', type=str, required=True, help='save weights to location')
    args = parser.parse_args()

    train_nn(args.dataset_root,args.save_weights_path)

    
    

