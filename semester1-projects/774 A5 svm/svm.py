import cvxpy as cp
import numpy as np
import pandas as pd
import json
import sys
import os
from sklearn.preprocessing import StandardScaler

C = 1
epsilon = 1e-4

def extract_substring(filename):
    """
    A new function created for extracting substring
    """
    basename = os.path.basename(filename)
    last_underscore = basename.rfind('_')
    last_dot = basename.rfind('.')
    if last_underscore != -1 and last_dot != -1:
        return basename[last_underscore + 1:last_dot]
    else:
        return None

def parse_data(filename):
    """
    get X and y
    """
    data = pd.read_csv(filename)
    features = data.iloc[:, :-1].values  
    target = data.iloc[:, -1].values
    y = np.where(target == 0, -1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    return X, y

def save_file(filename, data, indent=2):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent)


def svm(filename):
    X, y = parse_data(filename)
    n, d = X.shape

    w = cp.Variable(d)
    b = cp.Variable()
    xi = cp.Variable(n)
    
    objective = cp.Minimize(0.5 * cp.norm(w, 1) + C * cp.sum(xi))
    margin_constraint = cp.multiply(y, X @ w + b) >= 1 - xi
    slack_constraint = xi >= 0
    constraints = [margin_constraint, slack_constraint]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"] or np.any(xi.value > epsilon):
        separable = False
        support_vectors = []
    else:
        separable = True
        margins = y * (X @ w.value + b.value)
        support_vectors = np.where(np.abs(1 - margins) < epsilon)[0].tolist()

    
    weights = {"weights": w.value.tolist(), "bias": float(b.value)}
    sv = {"separable": int(separable), "support_vectors": support_vectors}

    output_prefix = extract_substring(filename)
    wtfile = f'weight_{output_prefix}.json'  
    svfile = f'sv_{output_prefix}.json' 
    
    save_file(wtfile, weights)
    save_file(svfile, sv)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
        
    filename = sys.argv[1]
    
    if not os.path.isfile(filename):
        print(f"File {filename} does not exist.")
        sys.exit(1)

    svm(filename)
