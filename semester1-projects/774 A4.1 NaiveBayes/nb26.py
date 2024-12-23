import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import argparse

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing (tokenization, stopword removal, lemmatization)
def preprocess_text(text):
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(processed_tokens)

# Normalize numeric columns
def normalize_numeric_columns(df, numeric_columns):
    df['numeric_sum'] = df[numeric_columns].sum(axis=1)
    for col in numeric_columns:
        df[col + '_ratio'] = df[col] / df['numeric_sum']
    df.drop(columns=['numeric_sum'], inplace=True)
    ratio_columns = [col + '_ratio' for col in numeric_columns]
    df[ratio_columns] = df[ratio_columns].fillna(0.5)
    return df, ratio_columns

# One-Hot Encoding manually using pandas
def one_hot_encode(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

# Logistic Regression from scratch using NumPy


def sigmoid(z):
    z = np.array(z, dtype=np.float64)  # Ensure z is always a NumPy array of floats
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.01, num_iter=1000):
    X = np.array(X, dtype=np.float64)  # Ensure X is a NumPy array of floats
    y = np.array(y, dtype=np.float64)  # Ensure y is a NumPy array of floats
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    for i in range(num_iter):
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # Gradient descent updates
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias

def predict(X, weights, bias):
    X = np.array(X, dtype=np.float64)  # Ensure X is a NumPy array
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return [1 if i > 0.5 else 0 for i in y_pred]



# Load and preprocess data
def load_and_preprocess_data(file_path, numeric_columns):
    df = pd.read_csv(file_path, sep='\t', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                     names=['label', 'sentence', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                            'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
                            'pants_on_fire_counts', 'context'])

    df, ratio_columns = normalize_numeric_columns(df, numeric_columns)
    df['processed_text'] = df['sentence'].apply(preprocess_text)

    return df, ratio_columns

# Manually split the data
def split_train_test(df, test_size=0.2):
    test_df = df.sample(frac=test_size, random_state=42)
    train_df = df.drop(test_df.index)
    return train_df, test_df

# Manual accuracy calculation
def accuracy_score_manual(y_true, y_pred):
    return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help="Train File Path")
    parser.add_argument('--test', type=str, help='Test File Path')
    parser.add_argument('--out', type=str, help='Output File Path for results')

    args = parser.parse_args()

    # Define numeric columns
    numeric_columns = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']

    # Load and preprocess datasets
    train_df, ratio_columns_train = load_and_preprocess_data(args.train, numeric_columns)
    valid_df, ratio_columns_valid = load_and_preprocess_data(args.test, numeric_columns)

    # One-Hot Encode categorical features
    categorical_features = ['subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'context']
    train_df = one_hot_encode(train_df, categorical_features)
    valid_df = one_hot_encode(valid_df, categorical_features)

    # Ensure both train and validation data have the same columns
    train_df, valid_df = train_df.align(valid_df, join='left', axis=1, fill_value=0)

    # Prepare the data for logistic regression
    X_train = train_df.drop(columns=['label', 'sentence', 'processed_text'])
    y_train = train_df['label'].apply(lambda x: 1 if x == 'true' else 0).values  # Binary classification: true vs false

    X_valid = valid_df.drop(columns=['label', 'sentence', 'processed_text'])
    y_valid = valid_df['label'].apply(lambda x: 1 if x == 'true' else 0).values

    # Train the logistic regression model
    weights, bias = logistic_regression(X_train.values, y_train, lr=0.01, num_iter=1000)

    # Make predictions on the validation set
    y_pred = predict(X_valid.values, weights, bias)

    # Calculate accuracy
    accuracy = accuracy_score_manual(y_valid, y_pred)

    # Save results to file
    with open(args.out, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")

    print(f"Test Accuracy: {accuracy:.4f}")
