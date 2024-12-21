import numpy as np
import argparse
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import ngrams

class MultinomialNBClassifier:
    def __init__(self, class_order):
        self.class_order = class_order
        self.priors = None
        self.conditional_probs_text = None
        self.conditional_probs_cat = None
        self.conditional_probs_num = None
        self.vocab = None

    def load_stopwords(self, file_path):
        """Loads stopwords from the provided file."""
        with open(file_path, 'r') as file:
            stopwords = set(line.strip() for line in file)
        return stopwords

    def standardize_features(self, X):
        """Standardizes numerical features."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (X - mean) / std

    def preprocess_text(self, text, stop_words):
        lemmatizer = WordNetLemmatizer()
        tokens = text.lower().split()
        return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    def extract_ngrams(self, tokens, n):
        """Extracts n-grams (e.g., unigrams, bigrams) from tokens."""
        return list(ngrams(tokens, n))

    def count_vectorizer_multinomial(self, corpus):
        """Generates a count vectorizer with unigrams and bigrams."""
        self.vocab = set()
        all_ngrams = []
        for sentence in corpus:
            unigrams = [(token,) for token in sentence]
            bigrams = self.extract_ngrams(sentence, 2)
            sentence_ngrams = unigrams + bigrams
            all_ngrams.append(sentence_ngrams)
            self.vocab.update(sentence_ngrams)
        self.vocab = sorted(self.vocab)

        vocab_map = {ngram: idx for idx, ngram in enumerate(self.vocab)}

        count_matrix = np.zeros((len(corpus), len(self.vocab)), dtype=int)
        for i, sentence in enumerate(all_ngrams):
            ngram_counts = {}
            for ngram in sentence:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            for ngram, count in ngram_counts.items():
                j = vocab_map[ngram]
                count_matrix[i, j] = count

        return count_matrix


    def handle_categorical_features(self, df, columns):
        """Encodes comma-separated categorical columns as binary features and handles missing values."""
        unique_values = {}
        binary_matrix = []

        # Handle each categorical column
        for col in columns:
            # Split comma-separated values, and replace NaN/empty with "<empty>"
            split_vals = df[col].fillna('<empty>').apply(lambda x: [v.strip().lower() for v in str(x).split(',')])

            # Get all unique categories
            unique_vals = sorted(set(val for sublist in split_vals for val in sublist))
            unique_values[col] = unique_vals

            # Create binary matrix for presence of categories
            binary_col_matrix = np.zeros((df.shape[0], len(unique_vals)))

            for i, vals in enumerate(split_vals):
                for val in vals:
                    if val in unique_vals:
                        binary_col_matrix[i, unique_vals.index(val)] = 1

            binary_matrix.append(binary_col_matrix)

        return np.hstack(binary_matrix), unique_values

    def calculate_class_priors(self, y_train):
        """Calculates the class priors P(y=j)."""
        priors = np.zeros(len(self.class_order))
        total_samples = y_train.shape[0]

        for idx, c in enumerate(self.class_order):
            class_count = np.sum(y_train == c)
            priors[idx] = class_count / total_samples

        return priors

    def calculate_conditional_probs(self, X_train, y_train, smoothing=1):
        """Calculates conditional probabilities P(feature_k | y=j) using Laplace smoothing."""
        num_classes = len(self.class_order)
        num_features = X_train.shape[1]
        conditional_probs = np.zeros((num_classes, num_features))

        for idx, c in enumerate(self.class_order):
            X_c = X_train[y_train == c]
            total_count = np.sum(X_c) + smoothing * num_features
            conditional_probs[idx] = (np.sum(X_c, axis=0) + smoothing) / total_count

        return conditional_probs


    def train(self, X_train_text, X_train_cat, X_train_num, y_train):
        """Trains the model by calculating priors and conditional probabilities for each feature type."""
        self.priors = self.calculate_class_priors(y_train)

        if X_train_text.shape[1] > 0:
            self.conditional_probs_text = self.calculate_conditional_probs(X_train_text, y_train)
        else:
            self.conditional_probs_text = None

        if X_train_cat.shape[1] > 0:
            self.conditional_probs_cat = self.calculate_conditional_probs(X_train_cat, y_train)
        else:
            self.conditional_probs_cat = None

        if X_train_num.shape[1] > 0:
            self.conditional_probs_num = self.calculate_conditional_probs(X_train_num, y_train)
        else:
            self.conditional_probs_num = None

    def eval(self, X_test_text, X_test_cat, X_test_num):
        """Evaluates the model on the test set."""
        log_priors = np.log(self.priors)
        num_samples = X_test_text.shape[0]
        log_posteriors = np.tile(log_priors, (num_samples, 1))

        if self.conditional_probs_text is not None:
            log_cond_probs_text = np.log(self.conditional_probs_text)
            log_likelihoods_text = X_test_text @ log_cond_probs_text.T
            log_posteriors += log_likelihoods_text

        if self.conditional_probs_cat is not None:
            log_cond_probs_cat = np.log(self.conditional_probs_cat)
            log_likelihoods_cat = X_test_cat @ log_cond_probs_cat.T
            log_posteriors += log_likelihoods_cat

        if self.conditional_probs_num is not None:
            log_cond_probs_num = np.log(self.conditional_probs_num)
            log_likelihoods_num = X_test_num @ log_cond_probs_num.T
            log_posteriors += log_likelihoods_num

        return np.exp(log_posteriors)

    def predict(self, X_test_text, X_test_cat, X_test_num):
        """Predicts labels for the test set."""
        proba_predictions = self.eval(X_test_text, X_test_cat, X_test_num)
        predicted_indices = np.argmax(proba_predictions, axis=1)
        predicted_labels = [self.class_order[i] for i in predicted_indices]
        return predicted_labels, proba_predictions

def simple_train_test_split(*arrays, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return [array[train_indices] for array in arrays] + [array[test_indices] for array in arrays]

def get_combinations(columns):
    """Generates all possible non-empty combinations of the input list."""
    combinations_list = []
    n = len(columns)
    for i in range(1, 1 << n):
        combo = []
        for j in range(n):
            if (i & (1 << j)):
                combo.append(columns[j])
        combinations_list.append(tuple(combo))
    return combinations_list

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total



def simple_train_test_split(*arrays, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return [array[train_indices] for array in arrays] + [array[test_indices] for array in arrays]

def evaluate_feature_combination(model, train_df, stopwords, text_col_idx, selected_categorical_columns, selected_numerical_columns):
    # Preprocess text data
    train_texts = train_df[text_col_idx].values
    train_labels = train_df[1].values
    train_texts = [model.preprocess_text(text, stopwords) for text in train_texts]

    # Prepare text features
    X_train_text = model.count_vectorizer_multinomial(train_texts)

    # Prepare categorical features
    if selected_categorical_columns:
        X_train_cat, _ = model.handle_categorical_features(train_df, list(selected_categorical_columns))
    else:
        X_train_cat = np.zeros((train_df.shape[0], 0))

    # Prepare numerical features
    if selected_numerical_columns:
        X_train_num = train_df[list(selected_numerical_columns)].values
        X_train_num = model.standardize_features(X_train_num)
    else:
        X_train_num = np.zeros((train_df.shape[0], 0))

    # Train-test split
    X_train_text_split, X_val_text_split, X_train_cat_split, X_val_cat_split, X_train_num_split, X_val_num_split, y_train_split, y_val_split = simple_train_test_split(
        X_train_text, X_train_cat, X_train_num, train_labels, test_size=0.2, random_state=42
    )

    # Train the model
    model.train(X_train_text_split, X_train_cat_split, X_train_num_split, y_train_split)

    # Evaluate on validation set
    predicted_labels, _ = model.predict(X_val_text_split, X_val_cat_split, X_val_num_split)
    accuracy = calculate_accuracy(y_val_split, predicted_labels)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=False)
    parser.add_argument("--out", type=str, required=False)
    parser.add_argument("--stop", type=str, required=True)
    args = parser.parse_args()

    # Initialize the model with class order
    model = MultinomialNBClassifier(class_order=["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"])

    # Load stopwords
    stopwords = model.load_stopwords(args.stop)

    # Load train dataset
    train_df = pd.read_csv(args.train, sep='\t', header=None)

    # Define columns
    text_col_idx = 2
    categorical_cols = [3, 4, 5, 6, 7, 13]
    numerical_cols = [8, 9, 10, 11, 12]

    best_accuracy = 0
    best_categorical_columns = []
    best_numerical_columns = []

    # Generate all possible combinations
    categorical_combinations = get_combinations(categorical_cols)
    numerical_combinations = get_combinations(numerical_cols)

    total_combinations = len(categorical_combinations) * len(numerical_combinations)
    combination_counter = 0

    for selected_categorical_columns in categorical_combinations:
        for selected_numerical_columns in numerical_combinations:
            combination_counter += 1
            if not selected_categorical_columns and not selected_numerical_columns:
                continue  # Skip if both are empty
            # Evaluate the current combination of columns
            accuracy = evaluate_feature_combination(
                model, train_df, stopwords, text_col_idx,
                selected_categorical_columns, selected_numerical_columns
            )

            print(f"Evaluating combination {combination_counter}/{total_combinations}: Categorical: {selected_categorical_columns}, Numerical: {selected_numerical_columns} -> Accuracy: {accuracy}")

            # Update the best combination if this improves accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_categorical_columns = selected_categorical_columns
                best_numerical_columns = selected_numerical_columns

    print(f"Best columns: Categorical: {best_categorical_columns}, Numerical: {best_numerical_columns} with accuracy: {best_accuracy}")
