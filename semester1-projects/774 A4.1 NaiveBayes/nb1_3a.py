import numpy as np
import argparse
import pandas as pd
from nltk.stem import PorterStemmer
from nltk import ngrams
import time

startTime = time.time()

class BernoulliNBClassifier:
    def __init__(self, class_order):
        self.class_order = class_order
        self.priors = None
        self.conditional_probs = None
        self.vocab = None

    def load_stopwords(self, file_path):
        """Loads stopwords from the provided file."""
        with open(file_path, 'r') as file:
            stopwords = set(line.strip() for line in file)
        return stopwords

    def preprocess_text(self, text, stop_words):
        """Removes stopwords and applies stemming."""
        ps = PorterStemmer()
        tokens = text.lower().split()
        processed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
        return processed_tokens

    def extract_ngrams(self, tokens, n):
        """Extracts n-grams (e.g., unigrams, bigrams) from tokens."""
        return list(ngrams(tokens, n))

    def count_vectorizer_binary(self, corpus, is_train=True):
        """Generates a binary count vectorizer with unigrams and bigrams."""
        all_ngrams = []
        for sentence in corpus:
            unigrams = [(token,) for token in sentence]
            bigrams = self.extract_ngrams(sentence, 2)
            sentence_ngrams = unigrams + bigrams
            all_ngrams.append(sentence_ngrams)
        
        if is_train:
            # Build the vocabulary from training data
            self.vocab = sorted(set(ngram for sentence in all_ngrams for ngram in sentence))
            vocab_map = {ngram: idx for idx, ngram in enumerate(self.vocab)}
            unknown_index = len(self.vocab)  # Index for unknown feature
            num_features = len(self.vocab) + 1  # Add one for unknown feature
        else:
            vocab_map = {ngram: idx for idx, ngram in enumerate(self.vocab)}
            unknown_index = len(self.vocab)  # Index for unknown feature
            num_features = len(self.vocab) + 1  # Add one for unknown feature

        binary_matrix = np.zeros((len(corpus), num_features), dtype=int)
        
        for i, sentence in enumerate(all_ngrams):
            unique_ngrams = set(sentence)  # To ensure binary presence
            has_unknown = False
            for ngram in unique_ngrams:
                if ngram in vocab_map:
                    binary_matrix[i, vocab_map[ngram]] = 1
                else:
                    has_unknown = True
            if has_unknown:
                binary_matrix[i, unknown_index] = 1  # Set unknown feature to 1 if any unseen n-grams

        return binary_matrix

    def calculate_class_priors(self, y_train):
        """Calculates the class priors P(y=j)."""
        priors = np.zeros(len(self.class_order))
        total_samples = y_train.shape[0]

        for idx, c in enumerate(self.class_order):
            class_count = np.sum(y_train == c)
            priors[idx] = class_count / total_samples

        return priors

    def calculate_conditional_probs(self, X_train, y_train):
        """Calculates the conditional probabilities P(feature_k = 1 | y=j) with Laplace smoothing."""
        num_classes = len(self.class_order)
        num_features = X_train.shape[1]
        conditional_probs = np.zeros((num_classes, num_features))

        for idx, c in enumerate(self.class_order):
            X_c = X_train[y_train == c]
            # Calculate P(feature_k = 1 | y = c)
            feature_counts = np.sum(X_c, axis=0)  # Sum over samples
            total_samples_c = X_c.shape[0]
            # Apply Laplace smoothing
            conditional_probs[idx, :] = (feature_counts + 1) / (total_samples_c + 2)

        return conditional_probs

    def train(self, X_train, y_train):
        """Trains the model by calculating priors and conditional probabilities."""
        self.priors = self.calculate_class_priors(y_train)
        self.conditional_probs = self.calculate_conditional_probs(X_train, y_train)

    def eval(self, X_test):
        """Evaluates the model on the test set."""
        num_samples = X_test.shape[0]
        num_classes = len(self.class_order)

        log_entry_matrix = np.zeros((num_samples, num_classes))

        for j in range(num_classes):
            prior = self.priors[j]
            # To prevent underflow, use log probabilities
            log_prior = np.log(prior)
            # Avoid log(0) by adding a small epsilon
            epsilon = 1e-10
            log_cond_probs = np.log(self.conditional_probs[j] + epsilon)
            log_cond_neg_probs = np.log(1 - self.conditional_probs[j] + epsilon)
            # Compute log likelihood
            log_likelihood = X_test @ log_cond_probs + (1 - X_test) @ log_cond_neg_probs
            log_entry_matrix[:, j] = log_prior + log_likelihood

        return log_entry_matrix

    def predict(self, X_test):
        """Predicts labels for the test set based on the maximum probability."""
        log_proba_predictions = self.eval(X_test)
        predicted_indices = np.argmax(log_proba_predictions, axis=1)
        predicted_labels = [self.class_order[i] for i in predicted_indices]
        return predicted_labels, log_proba_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stop", type=str, required=True)
    args = parser.parse_args()

    # Initialize the model with class order
    model = BernoulliNBClassifier(class_order=["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"])

    # Load stopwords and preprocess data
    stopwords = model.load_stopwords(args.stop)
    train_texts, train_labels = pd.read_csv(args.train, sep='\t', header=None, quoting=3)[[2, 1]].values.T
    train_texts = [model.preprocess_text(text, stopwords) for text in train_texts]
    test_texts, _ = pd.read_csv(args.test, sep='\t', header=None, quoting=3)[[2, 1]].values.T
    test_texts = [model.preprocess_text(text, stopwords) for text in test_texts]

    # Create binary count vectors using unigrams and bigrams
    X_train = model.count_vectorizer_binary(train_texts, is_train=True)
    X_test = model.count_vectorizer_binary(test_texts, is_train=False)

    # Train the model
    y_train = np.array(train_labels)
    model.train(X_train, y_train)

    # Make predictions
    predicted_labels, log_probas = model.predict(X_test)
    probas = np.exp(log_probas)  # Convert log probabilities back to probabilities

    # Write predictions to output file
    with open(args.out, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")

    # Compare with checker probabilities
    print(probas)
    data2 = np.load('./checker_files_v5/bernoulli_bigrams_probas_test.npy')
    print("Loaded checker probabilities:")
    print(data2)
    
    test_predicted_indices = np.argmax(data2, axis=1)
    predicted_indices = np.argmax(probas, axis=1)
    matches = np.sum(predicted_indices == test_predicted_indices)
    total = len(predicted_indices)
    percentage_match = (matches / total) * 100
    print(f"The percentage of matches between your script's argmax and the test probabilities is: {percentage_match:.2f}%")
    endTime = time.time()
    elapsed_time = endTime - startTime 
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Elapsed time: {int(minutes)} min {int(seconds)} sec")