import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk import ngrams
import argparse

class MultinomialNBClassifier:
    def __init__(self, class_order):
        self.class_order = class_order
        self.log_priors = None
        self.log_conditional_probs_text = None
        self.log_conditional_probs_cat = None
        self.log_conditional_probs_num = None
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
        return [ps.stem(word) for word in tokens if word not in stop_words]

    def extract_ngrams(self, tokens, n):
        """Extracts n-grams from tokens."""
        return list(ngrams(tokens, n))

    def handle_comma_separated_categorical(self, df, column):
        """Splits comma-separated categorical values and calculates probabilities."""
        split_categories = df[column].str.split(',')

        # Flatten the list of lists and make sure everything is lowercase and stripped of extra spaces
        unique_categories = sorted(set([cat.strip().lower() for sublist in split_categories.dropna().values for cat in sublist]))
        category_index = {cat: idx for idx, cat in enumerate(unique_categories)}

        # Pre-allocate the matrix
        count_matrix = np.zeros((len(df), len(unique_categories)), dtype=int)

        # Use efficient indexing to fill in the matrix
        for i, categories in enumerate(split_categories):
            if isinstance(categories, list):  # Check if the categories is a valid list
                for category in categories:
                    category = category.strip().lower()
                    if category in category_index:
                        count_matrix[i, category_index[category]] = 1  # Binary presence

        return count_matrix, unique_categories

    def count_vectorizer_multinomial(self, tokenized_sentences):
        """Generates a count vectorizer using tokenized sentences."""
        all_ngrams = []
        for sentence in tokenized_sentences:
            unigrams = [(token,) for token in sentence]
            bigrams = self.extract_ngrams(sentence, 2)
            all_ngrams.append(unigrams + bigrams)

        if self.vocab is None:
            self.vocab = sorted(set([ngram for sentence in all_ngrams for ngram in sentence]))
        vocab_index = {ngram: idx for idx, ngram in enumerate(self.vocab)}

        # Pre-allocate matrix
        count_matrix = np.zeros((len(tokenized_sentences), len(self.vocab)), dtype=int)

        # Vectorize using vocab dictionary lookup
        for i, sentence in enumerate(all_ngrams):
            for ngram in sentence:
                if ngram in vocab_index:
                    count_matrix[i, vocab_index[ngram]] += 1  # Count presence

        return count_matrix

    def calculate_class_priors(self, y_train):
        """Calculates class priors P(y=j) in log space."""
        # Map string labels to integer indices
        class_map = {label: idx for idx, label in enumerate(self.class_order)}
        
        # Convert y_train to integer labels using class_map
        y_train_mapped = np.array([class_map[label] for label in y_train])
        
        # Calculate priors using bincount
        priors = np.log(np.bincount(y_train_mapped, minlength=len(self.class_order)) / y_train_mapped.shape[0])
        
        return priors

    def calculate_conditional_probs(self, X_train, y_train):
        """Calculates conditional probabilities P(feature_k | y=j) in log space with Laplace smoothing."""
        num_classes = len(self.class_order)
        num_features = X_train.shape[1]
        log_conditional_probs = np.zeros((num_classes, num_features))

        for idx, c in enumerate(self.class_order):
            X_c = X_train[y_train == c]
            log_conditional_probs[idx, :] = np.log((np.sum(X_c, axis=0) + 1) / (np.sum(X_c) + num_features))

        return log_conditional_probs

    def prepare_features(self, df, stopwords, text_col_idx, categorical_cols, numeric_cols, train=True, cat_feature_maps=None):
        """Prepares text, categorical, and numeric features."""
        # Handle text features
        texts = df[text_col_idx].apply(lambda x: self.preprocess_text(x, stopwords))
        text_features = self.count_vectorizer_multinomial(texts)

        # Handle categorical features
        cat_features_list = []
        updated_cat_feature_maps = {}
        
        for col in categorical_cols:
            if train:  # For training data, generate the category map
                cat_features, unique_categories = self.handle_comma_separated_categorical(df, col)
                updated_cat_feature_maps[col] = unique_categories
            else:  # For test data, use the category map from training
                unique_categories = cat_feature_maps[col]
                split_categories = df[col].str.split(',')
                count_matrix = np.zeros((len(df), len(unique_categories)), dtype=int)
                category_index = {cat: idx for idx, cat in enumerate(unique_categories)}
                
                for i, categories in enumerate(split_categories):
                    if isinstance(categories, list):
                        for category in categories:
                            category = category.strip().lower()
                            if category in category_index:
                                count_matrix[i, category_index[category]] = 1  # Binary presence
                cat_features = count_matrix
            
            cat_features_list.append((col, cat_features, unique_categories))

        # Handle numeric features
        num_features = df[numeric_cols].values if len(numeric_cols) > 0 else np.zeros((len(df), 0))

        # Combine features
        combined_features = np.hstack([text_features] + [cat_features for _, cat_features, _ in cat_features_list] + [num_features])
        
        return combined_features, cat_features_list, text_features, num_features, updated_cat_feature_maps if train else None


    def train(self, X_train_text, X_train_cat, X_train_num, y_train):
        """Trains the Naive Bayes classifier."""
        self.log_priors = self.calculate_class_priors(y_train)
        self.log_conditional_probs_text = self.calculate_conditional_probs(X_train_text, y_train)
        self.log_conditional_probs_cat = self.calculate_conditional_probs(X_train_cat, y_train)
        self.log_conditional_probs_num = self.calculate_conditional_probs(X_train_num, y_train)

    def eval(self, X_test_text, X_test_cat, X_test_num):
        """Evaluates the classifier on the test set."""
        num_samples = X_test_text.shape[0]
        num_classes = len(self.class_order)
        entry_matrix = np.zeros((num_samples, num_classes))

        for j in range(num_classes):
            log_likelihood = self.log_priors[j]
            log_likelihood += np.dot(X_test_text, self.log_conditional_probs_text[j, :])
            log_likelihood += np.dot(X_test_cat, self.log_conditional_probs_cat[j, :])
            log_likelihood += np.dot(X_test_num, self.log_conditional_probs_num[j, :])
            entry_matrix[:, j] = log_likelihood

        return entry_matrix

    def predict(self, X_test_text, X_test_cat, X_test_num):
        """Predicts the class labels for the test set."""
        log_proba_predictions = self.eval(X_test_text, X_test_cat, X_test_num)
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

    # Initialize the model
    model = MultinomialNBClassifier(class_order=["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"])

    # Load stopwords
    stopwords = model.load_stopwords(args.stop)

    # Load train and test datasets
    train_df = pd.read_csv(args.train, sep='\t', header=None)
    test_df = pd.read_csv(args.test, sep='\t', header=None)

    text_col_idx = 2
    categorical_cols = [3, 4, 5, 6, 7, 13]
    numeric_cols = list(range(8, 12))

    # Prepare features for training set (store the category maps)
    X_train, cat_features_list_train, text_features_train, num_features_train, cat_feature_maps = model.prepare_features(
        train_df, stopwords, text_col_idx, categorical_cols, numeric_cols, train=True
    )

    # Prepare features for test set (use the category maps from the training set)
    X_test, cat_features_list_test, text_features_test, num_features_test, _ = model.prepare_features(
        test_df, stopwords, text_col_idx, categorical_cols, numeric_cols, train=False, cat_feature_maps=cat_feature_maps
    )

    # Now check if the number of features aligns
    n_text_features = len(model.vocab)
    n_cat_features_train = sum(len(unique) for _, _, unique in cat_features_list_train)  # Use training cat features
    n_num_features = len(numeric_cols)

    # Print out the feature sizes for debugging
    print(f"n_text_features: {n_text_features}")
    print(f"n_cat_features (training): {n_cat_features_train}")
    print(f"n_num_features: {n_num_features}")

    # Use this to slice X_train and X_test appropriately
    X_train_text = X_train[:, :n_text_features]
    X_train_cat = X_train[:, n_text_features:n_text_features + n_cat_features_train]
    X_train_num = X_train[:, n_text_features + n_cat_features_train:]

    X_test_text = X_test[:, :n_text_features]
    X_test_cat = X_test[:, n_text_features:n_text_features + n_cat_features_train]  # Use train's n_cat_features
    X_test_num = X_test[:, n_text_features + n_cat_features_train:]

    # Print the shapes for debugging
    print(f"X_train_text shape: {X_train_text.shape}")
    print(f"X_train_cat shape: {X_train_cat.shape}")
    print(f"X_train_num shape: {X_train_num.shape}")
    print(f"X_test_text shape: {X_test_text.shape}")
    print(f"X_test_cat shape: {X_test_cat.shape}")
    print(f"X_test_num shape: {X_test_num.shape}")

    # Train the model
    y_train = train_df[1].values
    model.train(X_train_text, X_train_cat, X_train_num, y_train)

    # Make predictions
    predicted_labels, _ = model.predict(X_test_text, X_test_cat, X_test_num)

    # Simulating accuracy check (replace with actual test labels if available)
    test_true_labels = test_df[1].values  # Assuming test labels are in column 1
    accuracy = np.mean(predicted_labels == test_true_labels)
    print(f"Accuracy: {accuracy:.2f}")

    # Write predictions to output file
    with open(args.out, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")
