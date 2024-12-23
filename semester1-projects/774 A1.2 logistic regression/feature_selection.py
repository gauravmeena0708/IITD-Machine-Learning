import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

EPS = np.float64(1e-12)

drop_cols = []
target_col = 'Gender'
binary_features = []
untouched_features = ['Total Costs', 'Length of Stay', 'Birth Weight', 'Operating Certificate Number']

severity_mapping = {
    1: 4,
    2: 3,
    3: 1,
    4: 2
}

map_features = {}

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, Y, weights):
    m = X.shape[0]
    Z = np.dot(X, weights)
    A = sigmoid(Z)
    return -np.sum(Y * np.log(A + EPS) + (1 - Y) * np.log(1 - A + EPS)) / m

def binary_pso(X_train, Y_train, n_particles=30, max_iter=50, n_features=1628, n_best_features=1000):
    particles = np.random.randint(2, size=(n_particles, n_features))
    velocities = np.random.uniform(-1, 1, (n_particles, n_features))
    personal_best = particles.copy()
    personal_best_loss = np.array([float('inf')] * n_particles)
    
    global_best = None
    global_best_loss = float('inf')
    w = 0.5
    c1 = 1.5
    c2 = 1.5

    for i in range(max_iter):
        for j in range(n_particles):
            fitness = fitness_function(X_train, Y_train, particles[j])
        
            if fitness < personal_best_loss[j]:
                personal_best_loss[j] = fitness
                personal_best[j] = particles[j].copy()
            
            if fitness < global_best_loss:
                global_best_loss = fitness
                global_best = particles[j].copy()
        
        for j in range(n_particles):
            velocities[j] = (w * velocities[j] +
                             c1 * np.random.random() * (personal_best[j] - particles[j]) +
                             c2 * np.random.random() * (global_best - particles[j]))
            
            sigmoid_velocity = sigmoid(velocities[j])
            particles[j] = np.where(sigmoid_velocity > 0.5, 1, 0)
        
        print(f"Iteration {i + 1}/{max_iter}, Best Loss: {global_best_loss}")
    
    selected_features = np.argsort(global_best)[-n_best_features:]
    
    return selected_features, global_best_loss

def fitness_function(X_train, Y_train, selected_features):
    selected_features = [index for index in selected_features if 0 <= index < X_train.shape[1]]

    if len(selected_features) == 0:
        return float('inf')

    X_train_selected = X_train[:, selected_features]
    _, finalloss = get_finalloss(X_train_selected, Y_train)

    return finalloss

def calculate_gradient(w, X, y, pred_probs):
    diff = (pred_probs - y)
    return (X.T @ diff) / X.shape[0]

def gradient_descent(X, y, W, base_rate, epochs, batch_size, beta1=0.9, beta2=0.99):
    m_t = np.zeros_like(W)
    v_t = np.zeros_like(W)

    for epoch_num in range(epochs):
        for start_idx in range(0, X.shape[0], batch_size):
            X_batch = X[start_idx:start_idx + batch_size]
            y_batch = y[start_idx:start_idx + batch_size]
            pred_probs = sigmoid(X_batch @ W)
            gradient = calculate_gradient(W, X_batch, y_batch, pred_probs)
            m_t = beta1 * m_t + (1 - beta1) * gradient
            v_t = beta2 * v_t + (1 - beta2) * (gradient ** 2)
            m_t_hat = m_t / (1 - beta1 ** (epoch_num + 1))
            v_t_hat = v_t / (1 - beta2 ** (epoch_num + 1))
            W -= base_rate * m_t_hat / (np.sqrt(v_t_hat) + 1e-16)

    return W, sigmoid(X @ W)

def get_finalloss(X_train, Y_train):
    learningRateList = [1]
    minLoss = float('inf')
    n_features = X_train.shape[1]

    finalW, finalSigmoid = None, None

    for learningRate in learningRateList:
        W = np.random.randn(n_features) / np.sqrt(n_features)
        W, _ = gradient_descent(X_train, Y_train, W, learningRate, 1, X_train.shape[0] // 10)
        loss = compute_loss(X_train, Y_train, W)
        
        if loss < minLoss:
            minLoss = loss
            finalW = W
    finalloss = compute_loss(X_train, Y_train, finalW)
    return finalW, finalloss

def feature_generation(df):
    df = df.drop(columns=drop_cols)

    label_encoder = LabelEncoder()
    for feature in binary_features:
        if feature in df.columns:
            df[feature] = label_encoder.fit_transform(df[feature])
    
    for col, mapping in map_features.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    oh_features = list(set(df.columns) - set(map_features.keys()) - set(drop_cols) - {target_col} - set(binary_features) - set(untouched_features))
    print(oh_features)
    if oh_features:
        ohe = OneHotEncoder(drop='first', sparse_output=False)
        df_ohe = ohe.fit_transform(df[oh_features])
        ohe_feature_names = ohe.get_feature_names_out(oh_features)
        df = df.drop(oh_features, axis=1)
        df = pd.concat([df, pd.DataFrame(df_ohe, columns=ohe_feature_names, index=df.index)], axis=1)

    return df

def get_data(train_file):
    df_X_train = pd.read_csv(train_file)
    df_Y_train = df_X_train[target_col].replace({-1: 0, 1: 1})
    df_X_train = df_X_train.drop(columns=[target_col])
    return df_X_train, df_Y_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('trainPath', type=str, help='Path to training file.')
    parser.add_argument('createdPath', type=str, help='Path to save created features.')
    parser.add_argument('selectedPath', type=str, help='Path to save selected features.')
    args = parser.parse_args()

    X_train, Y_train = get_data(args.trainPath)
    X_train_features = feature_generation(X_train)

    created_features = X_train_features.columns.to_numpy()
    np.savetxt(args.createdPath, created_features, fmt='%s')
    print("Saved Created features")

    n_particles = 1
    max_iter = 1
    selected_features, best_loss = binary_pso(X_train.values, Y_train.values, n_particles, max_iter)
    selected_values = np.zeros(len(created_features), dtype=int)
    # Set values to 1 for selected feature indices
    selected_values[selected_features-1] = 1
    np.savetxt(args.selectedPath, selected_values)
