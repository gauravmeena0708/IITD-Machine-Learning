import numpy as np
import argparse
import pandas as pd
import csv
from nltk.stem import PorterStemmer
from nltk import ngrams
from collections import Counter

class MultinomialNBClassifier:
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

    def count_vectorizer_multinomial(self, corpus):
        """Generates a count vectorizer with unigrams and bigrams for MultinomialNB."""
        all_ngrams = []
        for sentence in corpus:
            unigrams = [(token,) for token in sentence]  # Convert unigrams to tuples for consistency
            bigrams = self.extract_ngrams(sentence, 2)
            all_ngrams.append(unigrams + bigrams)
        
        if self.vocab is None:
            self.vocab = sorted(set(ngram for sentence in all_ngrams for ngram in sentence))
        
        # Create a dictionary for fast index lookup
        vocab_map = {ngram: idx for idx, ngram in enumerate(self.vocab)}

        # Preallocate count matrix
        count_matrix = np.zeros((len(corpus), len(self.vocab)), dtype=int)

        # Count n-grams using Counter
        for i, sentence in enumerate(all_ngrams):
            ngram_counts = Counter(sentence)
            for ngram, count in ngram_counts.items():
                if ngram in vocab_map:
                    idx = vocab_map[ngram]
                    count_matrix[i, idx] = count
        
        return count_matrix    

    def calculate_class_priors(self, y_train):
        """Calculates the class priors P(y=j)."""
        priors = np.zeros(len(self.class_order))
        total_samples = y_train.shape[0]

        for idx, c in enumerate(self.class_order):
            class_count = np.sum(y_train == c)
            priors[idx] = class_count / total_samples

        return priors

    def calculate_conditional_probs(self, X_train, y_train):
        """Calculates the conditional probabilities P(feature_k | y=j)."""
        num_classes = len(self.class_order)
        num_features = X_train.shape[1]
        conditional_probs = np.zeros((num_classes, num_features))

        for idx, c in enumerate(self.class_order):
            X_c = X_train[y_train == c]
            # For MultinomialNB, we calculate P(feature_k | y=j) as (count + 1) / (total words in class + num_features)
            feature_counts = np.sum(X_c, axis=0)
            total_counts = np.sum(feature_counts)
            conditional_probs[idx, :] = (feature_counts + 1) / (total_counts + num_features)

        return conditional_probs

    def train(self, X_train, y_train):
        """Trains the model by calculating priors and conditional probabilities."""
        self.priors = self.calculate_class_priors(y_train)
        self.conditional_probs = self.calculate_conditional_probs(X_train, y_train)

    def eval(self, X_test):
        """Evaluates the model on the test set."""
        log_priors = np.log(self.priors + 1e-15)
        log_cond_probs = np.log(self.conditional_probs + 1e-15)
        
        # Compute log joint likelihood
        log_likelihoods = X_test @ log_cond_probs.T
        log_posteriors = log_likelihoods + log_priors
        
        # Normalize probabilities using log-sum-exp
        max_log_posteriors = np.max(log_posteriors, axis=1, keepdims=True)
        log_probs = log_posteriors - (max_log_posteriors + np.log(np.sum(np.exp(log_posteriors - max_log_posteriors), axis=1, keepdims=True)))
        probabilities = np.exp(log_probs)
        
        return probabilities

    def predict(self, X_test):
        """Predicts labels for the test set based on the maximum probability."""
        proba_predictions = self.eval(X_test)
        predicted_indices = np.argmax(proba_predictions, axis=1)
        predicted_labels = [self.class_order[i] for i in predicted_indices]
        return predicted_labels, proba_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stop", type=str, required=True)
    args = parser.parse_args()

    # Initialize the model with class order
    model = MultinomialNBClassifier(class_order=["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"])

    # Load stopwords and preprocess data
    stopwords = model.load_stopwords(args.stop)
    train_df = pd.read_csv(args.train, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    train_texts = train_df[2].values
    train_labels = train_df[1].values
    train_texts = [model.preprocess_text(text, stopwords) for text in train_texts]

    test_df = pd.read_csv(args.test, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    test_texts = test_df[2].values
    test_texts = [model.preprocess_text(text, stopwords) for text in test_texts]

    # Create count vectors using unigrams and bigrams
    X_train = model.count_vectorizer_multinomial(train_texts)
    X_test = model.count_vectorizer_multinomial(test_texts)

    # Train the model
    y_train = np.array(train_labels)
    model.train(X_train, y_train)

    # Make predictions
    predicted_labels, probas = model.predict(X_test)

    # Write predictions to output file
    with open(args.out, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")

    # Compare with checker probabilities
    print(probas)
    data2 = np.load('./check5/multinomial_bigrams_probas_test.npy')
    print("Loaded checker probabilities:")
    print(data2)
    
    # Ensure class orders match
    # If necessary, rearrange 'probas' or 'data2' to match class orders

    test_predicted_indices = np.argmax(data2, axis=1)
    predicted_indices = np.argmax(probas, axis=1)
    matches = np.sum(predicted_indices == test_predicted_indices)
    total = len(predicted_indices)
    percentage_match = (matches / total) * 100
    print(f"The percentage of matches between your script's argmax and the test probabilities is: {percentage_match:.2f}%")
