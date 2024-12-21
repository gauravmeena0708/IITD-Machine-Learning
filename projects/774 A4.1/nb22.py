import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import argparse

# Download NLTK resources (ensure this is done before running the script)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = text.split()
    processed_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return processed_tokens  # Return a list of tokens

def generate_feature_combinations(features):
    # Generate all combinations of features without using itertools
    combinations = []
    n = len(features)
    for i in range(1, 1 << n):
        combo = [features[j] for j in range(n) if (i & (1 << j))]
        combinations.append(combo)
    return combinations

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def one_hot_encode(df, columns):
    # One-hot encode categorical variables using pandas get_dummies
    return pd.get_dummies(df[columns].astype(str), columns=columns)

def compute_tf_idf(corpus, vocab=None, idf=None):
    # Build vocabulary if not provided
    if vocab is None:
        vocab = {}
        for tokens in corpus:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
    # Initialize term frequency matrix
    tf = np.zeros((len(corpus), len(vocab)))
    for i, tokens in enumerate(corpus):
        for token in tokens:
            if token in vocab:
                tf[i, vocab[token]] += 1
    # Compute IDF if not provided
    if idf is None:
        df = np.sum(tf > 0, axis=0)
        idf = np.log((len(corpus) + 1) / (df + 1)) + 1  # Adding 1 to avoid division by zero
    # Compute TF-IDF
    tf_idf = tf * idf
    return tf_idf, vocab, idf

def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    X_standardized = (X - mean) / std
    return X_standardized

class MultinomialNBClassifier:
    def __init__(self, class_order):
        self.class_order = class_order
        self.priors = None
        self.conditional_probs = None

    def load_stopwords(self, file_path):
        """Loads stopwords from the provided file."""
        with open(file_path, 'r') as file:
            stopwords = set(line.strip() for line in file)
        return stopwords
        
    def calculate_class_priors(self, y_train):
        """Calculates the class priors P(y=j)."""
        priors = np.zeros(len(self.class_order))
        total_samples = y_train.shape[0]
        for idx, c in enumerate(self.class_order):
            class_count = np.sum(y_train == c)
            priors[idx] = class_count / total_samples
        return priors

    def calculate_conditional_probs(self, X_train, y_train, smoothing=1):
        """Calculates conditional probabilities P(feature_k | y=j) with Laplace smoothing."""
        num_classes = len(self.class_order)
        num_features = X_train.shape[1]
        conditional_probs = np.zeros((num_classes, num_features))
        for idx, c in enumerate(self.class_order):
            X_c = X_train[y_train == c]
            feature_counts = np.sum(X_c, axis=0)
            total_count = np.sum(feature_counts)
            conditional_probs[idx, :] = (feature_counts + smoothing) / (total_count + smoothing * num_features)
        return conditional_probs

    def train(self, X_train, y_train):
        self.priors = self.calculate_class_priors(y_train)
        self.conditional_probs = self.calculate_conditional_probs(X_train, y_train)

    def predict(self, X_test):
        log_priors = np.log(self.priors)
        # Adding a small value to avoid log(0)
        epsilon = 1e-10
        log_conditional_probs = np.log(self.conditional_probs + epsilon)
        log_likelihood = X_test @ log_conditional_probs.T
        log_posteriors = log_likelihood + log_priors
        predicted_indices = np.argmax(log_posteriors, axis=1)
        predicted_labels = [self.class_order[i] for i in predicted_indices]
        return np.array(predicted_labels)

def evaluate_feature_combinations(train_df, valid_df, combinations):
    results = []

    # Preprocess text data
    train_df['processed_text'] = train_df['sentence'].apply(preprocess_text)
    valid_df['processed_text'] = valid_df['sentence'].apply(preprocess_text)

    # Compute TF-IDF for training data
    X_train_text, vocab, idf = compute_tf_idf(train_df['processed_text'].tolist())
    # Compute TF-IDF for validation data using the same vocabulary and IDF
    X_valid_text, _, _ = compute_tf_idf(valid_df['processed_text'].tolist(), vocab=vocab, idf=idf)

    # Preprocess categorical features
    categorical_columns = ['subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'context']
    X_train_cat = one_hot_encode(train_df, categorical_columns).values
    X_valid_cat = one_hot_encode(valid_df, categorical_columns).reindex(columns=train_df.columns, fill_value=0).values

    # Process numeric features
    numeric_columns = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']
    X_train_num = train_df[numeric_columns].fillna(0).values
    X_valid_num = valid_df[numeric_columns].fillna(0).values

    X_train_num = standardize_features(X_train_num)
    X_valid_num = standardize_features(X_valid_num)

    # Combine all features into a dictionary
    feature_dict = {
        'text': (X_train_text, X_valid_text),
        'categorical': (X_train_cat, X_valid_cat),
        'numeric': (X_train_num, X_valid_num)
    }

    # Encode target labels to numeric
    y_train_labels, label_mapping = pd.factorize(train_df['label'])
    y_valid_labels = pd.Categorical(valid_df['label'], categories=label_mapping).codes

    # Handle class imbalance
    class_counts = np.bincount(y_train_labels)
    total_samples = len(y_train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_counts))}

    # Map numeric labels back to original labels
    class_order = label_mapping.tolist()

    # Loop over each combination of features
    for combination in combinations:
        print(f"Evaluating combination: {combination}")

        # Concatenate features for the current combination
        X_train_combined = np.hstack([feature_dict[feat][0] for feat in combination])
        X_valid_combined = np.hstack([feature_dict[feat][1] for feat in combination])

        # Train Multinomial Naive Bayes classifier
        clf = MultinomialNBClassifier(class_order=class_order)
        clf.train(X_train_combined, y_train_labels)
        y_pred = clf.predict(X_valid_combined)

        # Evaluate the model
        accuracy = calculate_accuracy(y_valid_labels, y_pred)
        print(f"Accuracy for {combination}: {accuracy:.4f}")

        # Store results
        results.append({'Combination': '+'.join(combination), 'Accuracy': accuracy})

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help="Train File Path")
    parser.add_argument('--test', type=str, help='Test File Path')
    parser.add_argument('--out', type=str, help='Output File Path for feature combination results')
    parser.add_argument("--stop", type=str, required=True)

    args = parser.parse_args()
    stopwords = model.load_stopwords(args.stop)
    
    # Load datasets
    column_names = ['label', 'sentence', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                    'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
                    'pants_on_fire_counts', 'context']

    train_df = pd.read_csv(args.train, sep='\t', header=None, usecols=range(1, 14), names=column_names)
    valid_df = pd.read_csv(args.test, sep='\t', header=None, usecols=range(1, 14), names=column_names)

    # Define feature groups
    features = ['text', 'categorical', 'numeric']

    # Generate all feature combinations
    feature_combinations = generate_feature_combinations(features)

    # Evaluate each feature combination and collect results
    results = evaluate_feature_combinations(train_df, valid_df, feature_combinations)

    # Save results to CSV (since xlsxwriter is not allowed)
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out, index=False)

    print(f"Results saved to {args.out}")
