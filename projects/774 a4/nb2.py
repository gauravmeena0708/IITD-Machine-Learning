import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class MultinomialNBClassifier:
    def __init__(self, class_order):
        self.class_order = class_order
        self.priors = None
        self.conditional_probs = None
        self.vocab = None

    def load_stopwords(self, file_path):
        with open(file_path, 'r') as file:
            stopwords = set(line.strip() for line in file)
        return stopwords

    def preprocess_text(self, text, stop_words):
        ps = PorterStemmer()
        tokens = text.lower().split()
        processed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
        
        bigrams = ['_'.join(bigram) for bigram in zip(processed_tokens, processed_tokens[1:])]
        
        return ' '.join(processed_tokens + bigrams)

    def count_vectorizer_multinomial(self, corpus, is_train=True):
        tokenized_sentences = [sentence.split() for sentence in corpus]

        if is_train:
            self.vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
            vocab_lookup = {word: idx for idx, word in enumerate(self.vocab)}  # Mapping vocab to indices
            unknown_index = len(self.vocab)  # Index for unknown words
        else:
            vocab_lookup = {word: idx for idx, word in enumerate(self.vocab)}
            unknown_index = len(self.vocab)

        num_features = len(self.vocab) + 1  # One extra for unknown words
        count_matrix = np.zeros((len(corpus), num_features), dtype=int)

        # Fill the count matrix
        for i, sentence in enumerate(tokenized_sentences):
            unseen_count = 0
            for word in sentence:
                if word in vocab_lookup:
                    count_matrix[i, vocab_lookup[word]] += 1
                else:
                    unseen_count += 1
            if unseen_count > 0:
                count_matrix[i, unknown_index] = unseen_count  # Store count of unseen words

        return count_matrix

    def calculate_class_priors(self, y_train):
        priors = np.zeros(len(self.class_order))
        total_samples = len(y_train)

        for idx, c in enumerate(self.class_order):
            priors[idx] = np.sum(y_train == c) / total_samples

        return priors

    def calculate_conditional_probs(self, X_train, y_train):
        num_classes = len(self.class_order)
        num_features = X_train.shape[1]
        conditional_probs = np.zeros((num_classes, num_features))

        for idx, c in enumerate(self.class_order):
            X_c = X_train[y_train == c]
            conditional_probs[idx, :] = (np.sum(X_c, axis=0) + 1) / (np.sum(X_c) + num_features)

        return conditional_probs

    def train(self, X_train, y_train):
        self.priors = self.calculate_class_priors(y_train)
        self.conditional_probs = self.calculate_conditional_probs(X_train, y_train)

    def eval(self, X_test):
        """Evaluates the model on the test set using log-space for efficiency."""
        log_priors = np.log(self.priors)  # Shape: (num_classes,)
        log_cond_probs = np.log(self.conditional_probs)  # Shape: (num_classes, num_features)
        
        # Calculate log-likelihoods: Shape: (num_samples, num_classes)
        log_likelihoods = X_test @ log_cond_probs.T  # X_test shape: (num_samples, num_features)

        # Add log priors to get log posteriors: Shape: (num_samples, num_classes)
        log_posteriors = log_likelihoods + log_priors  # Broadcasting log_priors

        # Convert log posteriors back to probabilities
        return np.exp(log_posteriors)  # Shape: (num_samples, num_classes)


    def predict(self, X_test):
        proba_predictions = self.eval(X_test)
        predicted_indices = np.argmax(proba_predictions, axis=1)
        predicted_labels = [self.class_order[i] for i in predicted_indices]
        return predicted_labels

# Utility functions
def one_hot_encode(data, unique_values):
    encoded_matrix = np.zeros((len(data), len(unique_values)))
    for i, val in enumerate(data):
        if val in unique_values:
            encoded_matrix[i, unique_values.index(val)] = 1
    return encoded_matrix

# Modify this function to handle comma-separated values
def one_hot_encode_comma_separated(data):
    # Convert all values to strings and fill missing values with empty string
    data = data.astype(str)
    data = np.where(data == 'nan', '', data)

    # Find all unique values across all rows (handle empty cells as well)
    unique_values = sorted(set([item.strip() for sublist in data for item in sublist.split(',') if item.strip()]))

    # Create one-hot encoded matrix
    encoded_matrix = np.zeros((len(data), len(unique_values)))

    for i, val in enumerate(data):
        items = [item.strip() for item in val.split(',') if item.strip()]  # Handle empty and comma-separated values
        for item in items:
            if item in unique_values:
                encoded_matrix[i, unique_values.index(item)] = 1
    
    return encoded_matrix

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Modify this function to use the same vocabulary for both train and test
def count_vectorizer_multinomial(self, corpus, is_train=True):
    tokenized_sentences = [sentence.split() for sentence in corpus]

    if is_train:
        # If training, build the vocabulary
        self.vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
        vocab_lookup = {word: idx for idx, word in enumerate(self.vocab)}  # Mapping vocab to indices
    else:
        # Use the existing vocabulary (from training) for the test data
        vocab_lookup = {word: idx for idx, word in enumerate(self.vocab)}

    num_features = len(self.vocab)  # Use the same number of features as in the training set
    count_matrix = np.zeros((len(corpus), num_features), dtype=int)

    # Fill the count matrix
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in vocab_lookup:
                count_matrix[i, vocab_lookup[word]] += 1
            # Ignore unseen words in the test set (don't include unknown word counts)

    return count_matrix

def count_vectorizer_multinomial(self, corpus, is_train=True, min_df=2, max_df=0.8):
    tokenized_sentences = [sentence.split() for sentence in corpus]

    if is_train:
        word_counts = {}
        for sentence in tokenized_sentences:
            for word in set(sentence):
                word_counts[word] = word_counts.get(word, 0) + 1
        
        total_docs = len(corpus)
        # Prune words by document frequency
        self.vocab = sorted([word for word, count in word_counts.items() if min_df <= count <= max_df * total_docs])

        vocab_lookup = {word: idx for idx, word in enumerate(self.vocab)}
    else:
        vocab_lookup = {word: idx for idx, word in enumerate(self.vocab)}

    count_matrix = np.zeros((len(corpus), len(self.vocab)), dtype=int)
    
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in vocab_lookup:
                count_matrix[i, vocab_lookup[word]] += 1

    return count_matrix


# Modify the one-hot encoding function to ensure consistent category handling
def one_hot_encode(data, unique_values):
    # Ensure the same unique values are used for both training and test sets
    encoded_matrix = np.zeros((len(data), len(unique_values)))
    for i, val in enumerate(data):
        if val in unique_values:
            encoded_matrix[i, unique_values.index(val)] = 1
    return encoded_matrix

# Modify this function to ensure consistency between training and test set one-hot encoding
def one_hot_encode_comma_separated(data, unique_values=None):
    # Convert all values to strings and fill missing values with empty string
    data = data.astype(str)
    data = np.where(data == 'nan', '', data)

    if unique_values is None:
        # Build the unique values if not provided (during training)
        unique_values = sorted(set([item.strip() for sublist in data for item in sublist.split(',') if item.strip()]))

    # Create one-hot encoded matrix
    encoded_matrix = np.zeros((len(data), len(unique_values)))

    for i, val in enumerate(data):
        items = [item.strip() for item in val.split(',') if item.strip()]  # Handle empty and comma-separated values
        for item in items:
            if item in unique_values:
                encoded_matrix[i, unique_values.index(item)] = 1

    return encoded_matrix, unique_values

# Modify the try_combinations function to ensure consistency in text and categorical encoding
def try_combinations(df_train, df_test, train_texts, test_texts, y_train, y_test, stopwords, model):
    combinations = [
        {"categorical": [3, 4, 5, 6, 7, 13], "numerical": [8, 9, 10, 11, 12]},
    ]
    
    best_accuracy = 0
    best_combination = None
    vocab = None  # Track the vocabulary built during training

    for combination in combinations:
        categorical_cols = combination["categorical"]
        numerical_cols = combination["numerical"]
        
        # One-hot encode categorical columns
        X_train_categorical = []
        X_test_categorical = []
        unique_values_dict = {}  # To store unique values for each column

        for col in categorical_cols:
            if col == 3:  # Handle comma-separated values for column 3
                X_train_cat, unique_values = one_hot_encode_comma_separated(df_train[:, col])
                X_test_cat, _ = one_hot_encode_comma_separated(df_test[:, col], unique_values)
                X_train_categorical.append(X_train_cat)
                X_test_categorical.append(X_test_cat)
            else:
                unique_values = sorted(set(df_train[:, col].astype(str)))
                X_train_categorical.append(one_hot_encode(df_train[:, col], unique_values))
                X_test_categorical.append(one_hot_encode(df_test[:, col], unique_values))

        if X_train_categorical:
            X_train_categorical = np.hstack(X_train_categorical)
            X_test_categorical = np.hstack(X_test_categorical)
        else:
            X_train_categorical = np.array([]).reshape(len(df_train), 0)
            X_test_categorical = np.array([]).reshape(len(df_test), 0)
        
        # Normalize numerical columns
        X_train_numerical = []
        X_test_numerical = []
        for col in numerical_cols:
            X_train_numerical.append(normalize(df_train[:, col].astype(float))[:, np.newaxis])
            X_test_numerical.append(normalize(df_test[:, col].astype(float))[:, np.newaxis])

        if X_train_numerical:
            X_train_numerical = np.hstack(X_train_numerical)
            X_test_numerical = np.hstack(X_test_numerical)
        else:
            X_train_numerical = np.array([]).reshape(len(df_train), 0)
            X_test_numerical = np.array([]).reshape(len(df_test), 0)
        
        # Preprocess text and combine all features
        X_train_text = model.count_vectorizer_multinomial(train_texts, is_train=True)
        X_test_text = model.count_vectorizer_multinomial(test_texts, is_train=False)
        
        X_train_combined = np.hstack([X_train_text, X_train_categorical, X_train_numerical])
        X_test_combined = np.hstack([X_test_text, X_test_categorical, X_test_numerical])
        
        # Ensure train and test data have consistent dimensions
        if X_train_combined.shape[1] != X_test_combined.shape[1]:
            print(f"Feature mismatch for combination: {combination}. Skipping...")
            continue
        
        # Train and evaluate model
        model.train(X_train_combined, y_train)
        predicted_labels = model.predict(X_test_combined)
        
        accuracy = np.mean(predicted_labels == y_test)
        print(f"Combination {combination} achieved accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_combination = combination
    
    print(f"\nBest combination: {best_combination} with accuracy: {best_accuracy:.4f}")


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stop", type=str, required=True)
    args = parser.parse_args()

    model = MultinomialNBClassifier(class_order=["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"])
    stopwords = model.load_stopwords(args.stop)
    
    # Load train and test data
    df_train = pd.read_csv(args.train, sep='\t', header=None, quoting=3).values
    df_test = pd.read_csv(args.test, sep='\t', header=None, quoting=3).values
    train_texts = [model.preprocess_text(text, stopwords) for text in df_train[:, 2]]
    test_texts = [model.preprocess_text(text, stopwords) for text in df_test[:, 2]]
    
    y_train = df_train[:, 1]
    y_test = df_test[:, 1]
    
    try_combinations(df_train, df_test, train_texts, test_texts, y_train, y_test, stopwords, model)
