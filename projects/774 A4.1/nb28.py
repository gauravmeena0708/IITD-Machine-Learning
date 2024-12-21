import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
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
    return processed_tokens  # Return list of tokens for n-gram processing

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

# Function to calculate n-grams (uni-grams and bi-grams)
def generate_ngrams(tokens, n=1):
    return zip(*[tokens[i:] for i in range(n)])

import numpy as np
import pandas as pd
from collections import defaultdict

def generate_ngrams(tokens, n=1):
    return zip(*[tokens[i:] for i in range(n)])

def extract_text_features(df, ngram_range=(1, 2)):
    # Dictionary to hold the total counts of each n-gram
    ngram_features = defaultdict(int)
    feature_list = []

    # First pass: Collect n-gram counts and create feature list for each sentence
    for sentence in df['processed_text']:
        sentence_features = defaultdict(int)
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngrams = generate_ngrams(sentence, n)
            for ngram in ngrams:
                ngram_str = ' '.join(ngram)
                sentence_features[ngram_str] += 1
                ngram_features[ngram_str] += 1
        feature_list.append(sentence_features)

    # Get the sorted list of all n-gram features (columns for the matrix)
    all_ngrams = sorted(ngram_features.keys())
    num_ngrams = len(all_ngrams)

    # Initialize the feature matrix with zeros (dense matrix)
    num_sentences = len(df)
    feature_matrix = np.zeros((num_sentences, num_ngrams), dtype=np.float64)

    # Second pass: Fill the feature matrix with n-gram counts
    for i, sentence_features in enumerate(feature_list):
        for ngram, count in sentence_features.items():
            if ngram in all_ngrams:
                col_index = all_ngrams.index(ngram)
                feature_matrix[i, col_index] = count

    return feature_matrix, all_ngrams


# Naive Bayes training
def train_naive_bayes(X, y):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    
    # Initialize counts and probabilities
    class_counts = np.zeros(n_classes)
    feature_counts = np.zeros((n_classes, n_features))
    total_feature_counts = np.zeros(n_classes)

    for i, c in enumerate(classes):
        X_c = X[y == c]
        class_counts[i] = X_c.shape[0]
        feature_counts[i, :] = X_c.sum(axis=0)
        total_feature_counts[i] = feature_counts[i, :].sum()

    # Calculate log probabilities
    class_log_probs = np.log(class_counts / n_samples)
    feature_log_probs = np.log((feature_counts + 1) / (total_feature_counts[:, None] + n_features))

    return class_log_probs, feature_log_probs, classes

# Naive Bayes prediction
def predict_naive_bayes(X, class_log_probs, feature_log_probs, classes):
    log_probs = X @ feature_log_probs.T + class_log_probs
    return classes[np.argmax(log_probs, axis=1)]

# Load and preprocess data
def load_and_preprocess_data(file_path, numeric_columns):
    df = pd.read_csv(file_path, sep='\t', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                     names=['label', 'sentence', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                            'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
                            'pants_on_fire_counts', 'context'])

    df, ratio_columns = normalize_numeric_columns(df, numeric_columns)
    df['processed_text'] = df['sentence'].apply(preprocess_text)

    return df, ratio_columns

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

    # Extract text features using uni-grams and bi-grams
    X_train_text, ngram_features = extract_text_features(train_df)
    X_valid_text, _ = extract_text_features(valid_df)

    # Prepare numeric and categorical data
    X_train = pd.concat([train_df.drop(columns=['label', 'sentence', 'processed_text']), X_train_text], axis=1).fillna(0)
    X_valid = pd.concat([valid_df.drop(columns=['label', 'sentence', 'processed_text']), X_valid_text], axis=1).fillna(0)

    # Prepare labels
    y_train = train_df['label'].apply(lambda x: 1 if x == 'true' else 0).values
    y_valid = valid_df['label'].apply(lambda x: 1 if x == 'true' else 0).values

    # Train the Naive Bayes model
    class_log_probs, feature_log_probs, classes = train_naive_bayes(X_train.values, y_train)

    # Make predictions on the validation set
    y_pred = predict_naive_bayes(X_valid.values, class_log_probs, feature_log_probs, classes)

    # Calculate accuracy
    accuracy = accuracy_score_manual(y_valid, y_pred)

    # Save results to file
    with open(args.out, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")

    print(f"Test Accuracy: {accuracy:.4f}")
