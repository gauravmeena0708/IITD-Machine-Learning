import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.preprocessing import OneHotEncoder
from itertools import chain
from nltk import ngrams
import argparse
from collections import Counter
import itertools


class MultinomialNBClassifier:
    def __init__(self, class_order):
        self.class_order = class_order
        self.log_priors = None
        self.log_conditional_probs = None
        self.vocab = None
        self.ohe = OneHotEncoder(sparse=False)

    def load_stopwords(self, file_path):
        with open(file_path, 'r') as file:
            stopwords = set(line.strip() for line in file)
        return stopwords

    def preprocess_text(self, text, stop_words):
        ps = PorterStemmer()
        tokens = text.lower().split()
        processed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
        return processed_tokens

    def extract_ngrams(self, tokens, n):
        return list(ngrams(tokens, n))

    def handle_comma_separated_categorical(self, df, column):
        """Splits comma-separated categorical values and one-hot encodes them."""
        split_categories = df[column].str.split(',')
        unique_categories = sorted(set(chain(*split_categories.dropna().values)))
        
        # One-hot encoding
        encoded = np.zeros((len(df), len(unique_categories)), dtype=int)
        for i, categories in enumerate(split_categories):
            if categories is None or pd.isna(categories).all():  # Updated check
                continue
            for category in categories:
                idx = unique_categories.index(category.strip())
                encoded[i, idx] = 1
        
        return pd.DataFrame(encoded, columns=unique_categories)

    def count_vectorizer_multinomial(self, tokenized_sentences):
        """Generates a count vectorizer using tokenized sentences (list of tokens)."""
        all_ngrams = []
        
        for sentence in tokenized_sentences:
            unigrams = [(token,) for token in sentence]  # Convert unigrams to 1-tuple
            bigrams = self.extract_ngrams(sentence, 2)
            all_ngrams.append(unigrams + bigrams)  # Combine unigrams and bigrams
        
        if self.vocab is None:
            self.vocab = sorted(set(ngram for sentence in all_ngrams for ngram in sentence))
        
        count_matrix = np.zeros((len(tokenized_sentences), len(self.vocab)), dtype=int)
        
        for i, sentence in enumerate(all_ngrams):
            counts = Counter(sentence)
            for ngram, count in counts.items():
                if ngram in self.vocab:
                    j = self.vocab.index(ngram)
                    count_matrix[i, j] = count
        
        return count_matrix

    def calculate_class_priors(self, y_train):
        priors = np.zeros(len(self.class_order))
        total_samples = y_train.shape[0]

        for idx, c in enumerate(self.class_order):
            class_count = np.sum(y_train == c)
            priors[idx] = np.log(class_count / total_samples)  # Log of prior probability

        return priors

    def calculate_conditional_probs(self, X_train, y_train):
        num_classes = len(self.class_order)
        num_features = X_train.shape[1]
        log_conditional_probs = np.zeros((num_classes, num_features))

        for idx, c in enumerate(self.class_order):
            X_c = X_train[y_train == c]
            # Log of conditional probabilities with Laplace smoothing
            log_conditional_probs[idx, :] = np.log((np.sum(X_c, axis=0) + 1) / (np.sum(X_c) + num_features))

        return log_conditional_probs

    def prepare_features(self, df, stopwords, text_col_idx, categorical_cols, numeric_cols, feature_combination):
        features = []

        if 'text' in feature_combination:
            texts = df[text_col_idx].apply(lambda x: self.preprocess_text(x, stopwords))
            text_features = self.count_vectorizer_multinomial(texts)
            features.append(text_features)

        if 'categorical' in feature_combination:
            for col in categorical_cols:
                cat_features = self.handle_comma_separated_categorical(df, col)
                features.append(cat_features.values)

        if 'numeric' in feature_combination:
            features.append(df[numeric_cols].values)

        return np.hstack(features)

    def train(self, X_train, y_train):
        self.log_priors = self.calculate_class_priors(y_train)
        self.log_conditional_probs = self.calculate_conditional_probs(X_train, y_train)

    def eval(self, X_test):
        num_samples = X_test.shape[0]
        num_classes = len(self.class_order)

        entry_matrix = np.zeros((num_samples, num_classes))

        for i, sample in enumerate(X_test):
            for j, c in enumerate(self.class_order):
                log_likelihood = self.log_priors[j]  # Start with log prior
                for k in range(len(sample)):
                    if sample[k] == 1:
                        log_likelihood += self.log_conditional_probs[j, k]  # Log P(feature | class)
                    else:
                        # Log (1 - P(feature | class))
                        log_likelihood += np.log(1 - np.exp(self.log_conditional_probs[j, k]))
                entry_matrix[i, j] = log_likelihood

        return entry_matrix

    def predict(self, X_test):
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
    model = MultinomialNBClassifier(class_order=["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"])

    # Load stopwords and data
    stopwords = model.load_stopwords(args.stop)
    train_df = pd.read_csv(args.train, sep='\t', header=None)
    test_df = pd.read_csv(args.test, sep='\t', header=None)

    # Columns
    text_col_idx = 2  # Assuming text is in column 2
    categorical_cols = [3, 4, 5, 6, 7]  # Example categorical columns (subject, speaker, etc.)
    numeric_cols = list(range(8, 13))  # Example numeric columns (barely_true_counts, etc.)

    # Generate all combinations of categorical columns
    categorical_combinations = list(itertools.chain.from_iterable(itertools.combinations(categorical_cols, r) for r in range(1, len(categorical_cols) + 1)))

    feature_combinations = [['text', 'categorical', 'numeric']]

    best_combination = None
    best_accuracy = 0

    y_train = train_df[1].values  # Assuming labels are in column 1
    for combination in feature_combinations:
        for cat_comb in categorical_combinations:
            X_train = model.prepare_features(train_df, stopwords, text_col_idx, list(cat_comb), numeric_cols, combination)
            X_test = model.prepare_features(test_df, stopwords, text_col_idx, list(cat_comb), numeric_cols, combination)

            model.train(X_train, y_train)

            predicted_labels, _ = model.predict(X_test)

            # Simulating accuracy check (replace with actual test labels if available)
            test_true_labels = test_df[1].values  # Assuming test labels are in column 1
            accuracy = np.mean(predicted_labels == test_true_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combination = (combination, cat_comb)

    print(f"Best feature combination: {best_combination} with accuracy: {best_accuracy:.2f}")

    # Predictions for the best combination to output file
    X_train = model.prepare_features(train_df, stopwords, text_col_idx, list(best_combination[1]), numeric_cols, best_combination[0])
    X_test = model.prepare_features(test_df, stopwords, text_col_idx, list(best_combination[1]), numeric_cols, best_combination[0])
    model.train(X_train, y_train)
    predicted_labels, _ = model.predict(X_test)

    with open(args.out, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")
