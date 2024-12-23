import pandas as pd
import numpy as np

train_path = "train1.csv"
test_path  = "test1.csv"
pred_path  = "test_pred1.csv"
param_path = ".\\tests\\test1\\params.txt" 

def get_train_data(train_file):
    df = pd.read_csv(train_file)
    X0 = df.iloc[:, :-1].values
    
    # Add bias term (column of ones)
    X_train = np.hstack([np.ones((X0.shape[0], 1)), X0])
    
    # Extract labels and adjust class labels to start from 0
    Y_train = df.iloc[:, -1].values
    Y_train = Y_train - 1  # Assuming labels start from 1, we subtract 1 to start from 0
    
    # One-hot encoding of labels
    num_class = len(np.unique(Y_train))
    Y_train_oh = np.eye(num_class)[Y_train]

    return X_train, Y_train, Y_train_oh

def get_test_data(test_file, pred_file):
    X0 = pd.read_csv(test_file).values
    X_test = np.hstack([np.ones((X0.shape[0], 1)), X0])  # Adding bias
    Y_test = pd.read_csv(pred_file).iloc[:, -1].values
    Y_test = Y_test - 1  # Adjust class labels to start from 0
    
    # One-hot encoding of labels
    num_class = len(np.unique(Y_test))
    Y_test_oh = np.eye(num_class)[Y_test]
    
    return X_test, Y_test, Y_test_oh

def get_params(param_file):
    with open(param_file, 'r') as file:
        lines = file.readlines()
    
    # Read parameters from the file
    learning_strategy = np.int32(lines[0].strip())  # First line as an integer
    second_line_values = np.array(lines[1].strip().split(','), dtype=np.float64)
    
    if len(second_line_values) == 1:
        n0 = second_line_values[0]
        k = None  
    else:
        n0, k = second_line_values
    
    epochs = np.int32(lines[2].strip())
    batch_size = np.int32(lines[3].strip())
    
    return learning_strategy, n0, k , epochs, batch_size

def feature_normalize(X):
    num_features = X.shape[1]
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Load training data
X_train, Y_train_indices, Y_train_oh = get_train_data(train_path)
#X_train, mu, sigma = feature_normalize(X_train)
# Number of features and classes
n_features = X_train.shape[1]
num_class = len(np.unique(Y_train_indices))

# Initialize weights
#W = np.zeros((n_features, num_class)) 

# Load parameters
strategy, step, k, epochs, batch_size = get_params(param_path)
print(strategy, step, k, epochs, batch_size)

# Calculate the frequency of each class using the original class labels
freq = np.bincount(Y_train_indices, minlength=num_class)
print("Class Frequencies:", freq)

import numpy as np
#X_train = feature_normalize(X_train)
W = np.zeros((n_features, num_class)) 
#W = np.random.randn(n_features, num_class) * 0.01
def softmax(x, w):
    z = x @ w  # Shape (n, k)
    exp_z = np.exp(z-np.max(z,axis=1,keepdims=True))
    softmax_probs = exp_z / (np.sum(exp_z, axis=1, keepdims=True))
    return softmax_probs


def loss_function(w, X, y, freq):
    n, d = X.shape
    k = w.shape[1]
    loss = 0.0
    pred_probs = softmax(X, w)
    eps = 1e-15  # Small epsilon value
    pred_probs = np.clip(pred_probs, eps, 1 - eps)
    
    for i in range(n):
        for j in range(k):
            if y[i, j] == 1:
                loss -= np.log(pred_probs[i, j]) / freq[j]
    
    return loss / (2 * n)



def calculate_gradient(X, Y, W, freq):
    m = X.shape[0]
    k = W.shape[1]

    Y_pred = softmax(X, W)

    dz = Y_pred - Y
    dw = np.zeros_like(W)
    for j in range(k):
        dw[:, j] = (X.T @ (dz[:, j] / freq[j])) / (2 * m)

    return dw



learning_rate = 1.e-9

# Calculate the initial loss
for _ in range(epochs):
    loss_old = loss_function(W, X_train, Y_train_oh, freq)
    print("Loss before updating weights:", loss_old)
    gradient = calculate_gradient(X_train, Y_train_oh, W, freq)
    Wnew = W - (learning_rate * gradient)
    loss_new = loss_function(Wnew, X_train, Y_train_oh, freq)
    if loss_new<loss_old:
        W = Wnew
        print("New Ws:", Wnew[0])
        print("Loss after updating weights:", loss_new)
    else:
        learning_rate *=10
    
    


