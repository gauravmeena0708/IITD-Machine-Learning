import json
import sys

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_max_difference(data1, data2):
    weights1, weights2 = data1['weights'], data2['weights']
    bias1, bias2 = data1['bias'], data2['bias']

    # Check that both weight lists have the same length
    if len(weights1) != len(weights2):
        raise ValueError("The weight arrays in the two JSON files have different lengths.")
    
    # Calculate maximum difference in weights
    max_weight_diff = max(abs(w1 - w2) for w1, w2 in zip(weights1, weights2))
    
    # Calculate difference in biases
    bias_diff = abs(bias1 - bias2)
    
    # Print results
    print(f"Maximum weight difference: {max_weight_diff}")
    print(f"Bias difference: {bias_diff}")
    
    # Check if the differences are within the threshold
    threshold = 1e-4
    if max_weight_diff < threshold and bias_diff < threshold:
        print("All differences are within the acceptable threshold.")
    else:
        print("Differences exceed the acceptable threshold.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_json_diff.py <file1.json> <file2.json>")
        sys.exit(1)
    
    file1, file2 = sys.argv[1], sys.argv[2]
    data1 = load_json(file1)
    data2 = load_json(file2)
    
    calculate_max_difference(data1, data2)
