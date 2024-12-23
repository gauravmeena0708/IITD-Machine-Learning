import pandas as pd
import numpy as np
import argparse
import time

EPS = np.float64(1e-12)

def get_train_data(train_file):
    df = pd.read_csv(train_file)
    X0 = df.iloc[:, :-1].values
    X_train = np.hstack([np.ones((X0.shape[0], 1)), X0])
    Y_train = process_y(df.iloc[:, -1].values)
    return X_train, Y_train

def process_y(Y):
    Y = Y - 1
    num_classes = len(np.unique(Y))
    return np.eye(num_classes)[Y]

def get_params(param_file):
    with open(param_file, 'r') as file:
        lines = file.readlines()
    
    learning_strategy = int(lines[0].strip())
    second_line_values = np.array(lines[1].strip().split(','), dtype=np.float64)
    n0, k = second_line_values if len(second_line_values) > 1 else (second_line_values[0], None)
    epochs = int(lines[2].strip())
    batch_size = int(lines[3].strip())
    
    return learning_strategy, n0, k, epochs, batch_size

def ternary_search(base_rate, w, X, y, freq, max_iter=20):
    low, high = 0, base_rate
    gradient = calculate_gradient(w, X, y, freq, softmax(X, w))
    
    while loss_function(X, y, freq, softmax(X, w)) > loss_function(X, y, freq, softmax(X, w - high * gradient)):
        high *= 2
        if high > 1e3:
            break
    
    for i in range(max_iter):
        rate_1 = (2 * low + high) / 3
        rate_2 = (2 * high + low) / 3

        loss1 = loss_function(X, y, freq, softmax(X, w - rate_1 * gradient))
        loss2 = loss_function(X, y, freq, softmax(X, w - rate_2 * gradient))
        
        if loss1 < loss2:
            high = rate_2
        elif loss1 > loss2:
            low = rate_1
        else:
            low = rate_1
            high = rate_2
        
    return (low + high) / 2

def softmax(X, w):
    temp = X @ w 
    z = temp - np.max(temp, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return np.clip(exp_z / np.sum(exp_z, axis=1, keepdims=True), EPS, 1.0 - EPS)

def loss_function(X, y, freq, pred_probs):
    mask = y == 1
    weighted_log_likelihood = np.log(pred_probs[mask] + 1e-64) / freq[mask]
    return -np.sum(weighted_log_likelihood) / (2 * X.shape[0])

def calculate_gradient(w, X, y, freq, pred_probs):
    diff = (pred_probs - y) / freq
    return (X.T @ diff) / (2 * X.shape[0])

def get_freq(Y_train):
    freq_j = np.sum(Y_train, axis=0)
    return np.repeat((Y_train * freq_j).sum(axis=1).reshape(-1, 1), Y_train.shape[1], axis=1)

def gradient_descent(X, y, W, base_rate, epochs, batch_size, freq, strategy, k, beta1=0.85, beta2=0.9999):
    if strategy == 1:
        rate = base_rate

    gradientSum = np.zeros_like(W) 
    squaredGradientSum = np.zeros_like(W) 
    previousLoss = float('inf')
    for epoch_num in range(epochs):
        if strategy == 2:
            rate = base_rate / (1 + (k * (epoch_num+1)))

        for start_idx in range(0, X.shape[0], batch_size):
            
            X_batch = X[start_idx:start_idx + batch_size]
            y_batch = y[start_idx:start_idx + batch_size]
            freq_batch = freq[start_idx:start_idx + batch_size]
            
            if strategy ==3:
                rate = ternary_search(base_rate, W, X_batch, y_batch, freq_batch)
            pred_probs = softmax(X_batch, W) 
                
            gradient = calculate_gradient(W, X_batch, y_batch, freq_batch, pred_probs)

            if strategy != 4:
                W -= rate * gradient            
            else:
                gradientSum = beta1 * gradientSum + (1 - beta1) * gradient
                squaredGradientSum = beta2 * squaredGradientSum + (1 - beta2) * (gradient ** 2)

                gradientSumHat = gradientSum / (1 - beta1 ** (epoch_num+1))
                squaredGradientSumHat = squaredGradientSum / (1 - beta2 ** (epoch_num+1))

                W -= base_rate * gradientSumHat / (np.sqrt(squaredGradientSumHat) + 1e-64)
       
        if epoch_num % 50 == 0:
            print(f'Iteration {epoch_num}, Loss: {loss_function(X,y,freq,softmax(X,W))}')

    return W, softmax(X, W)


def preprocessing(X_train, X_test):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1)
    X_test = scaler.transform(X_test)
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)

    return X_train, X_test


def get_data(trainPath,testPath):
 
    df_X_train = pd.read_csv(trainPath)
    df_X_test = pd.read_csv(testPath)

    df_Y_train = df_X_train['Race'] 
    df_X_train = df_X_train.drop(columns=['Race'])

    return df_X_train, df_X_test, df_Y_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('problemType', type=str, choices=["a", "b"], help="Problem type")
    parser.add_argument('trainingFilePath', type=str, help='Path to training file.')
    parser.add_argument('inputFilePath', type=str, help='Path to test or params file')
    parser.add_argument('weightFilePath', type=str, help='Path to weight file.')
    parser.add_argument('predFilePath', type=str, nargs='?', default=None, help='Path to predictions output file (only for b).')

    args = parser.parse_args()

    if args.problemType == 'a':
        strategy, base_rate, k, epochs, batch_size = get_params(args.inputFilePath)
        X_train, Y_train = get_train_data(args.trainingFilePath)
        n_features, num_class = X_train.shape[1], Y_train.shape[1]
        W = np.zeros((n_features, num_class), dtype=np.float64)
        freq = get_freq(Y_train)

        print(f"Running {epochs} epochs, batch size: {batch_size}, base rate: {base_rate}")
        W,_ = gradient_descent(X_train, Y_train, W, base_rate, epochs, batch_size, freq, strategy, k)
        np.savetxt(args.weightFilePath, W.flatten(), delimiter=',')

    elif args.problemType == 'b':


        X_train, X_test, Y_train = get_data(args.trainingFilePath, args.inputFilePath)
        print("data loaded successfully")
        X_train , X_test = preprocessing(X_train, X_test)
        Y_train = pd.get_dummies(Y_train).astype(int).values
    
        freq = get_freq(Y_train)

        n_classes = Y_train.shape[1]
        n_features = X_train.shape[1]
        learningRateList = [0.001]
        minLoss = float('inf')
        
        for learningRate in learningRateList:
            print(f"calculating gradient descent for learning rate: {learningRate}")
            W = np.random.randn(n_features, n_classes) / np.sqrt(n_features)
            W, softmaxValue = gradient_descent(X_train, Y_train, W, learningRate, 250, X_train.shape[0]//4, freq, 4, 0)

            loss = loss_function(X_train, Y_train, freq, softmaxValue )
            if  loss < minLoss:
                minLoss = loss
                finalW = W
                finalSoftmax = softmaxValue


        testSoftmax = softmax(X_test, finalW)

        np.savetxt(args.weightFilePath, finalW, delimiter=',')
        np.savetxt(args.predFilePath, testSoftmax, delimiter=',')

    else:
        print("Error: Incorrect arguments or missing prediction file path.")
