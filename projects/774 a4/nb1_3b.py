import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

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
        
        # Create bigrams and combine them with unigrams
        bigrams = ['_'.join(bigram) for bigram in zip(processed_tokens, processed_tokens[1:])]
        
        # Combine unigrams and bigrams
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
        return predicted_labels, proba_predictions


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
    train_texts, train_labels = pd.read_csv(args.train, sep='\t', header=None, quoting=3)[[2, 1]].values.T
    train_texts = [model.preprocess_text(text, stopwords) for text in train_texts]
    test_texts, _ = pd.read_csv(args.test, sep='\t', header=None, quoting=3)[[2, 1]].values.T
    test_texts = [model.preprocess_text(text, stopwords) for text in test_texts]
    
    X_train = model.count_vectorizer_multinomial(train_texts, is_train=True)
    X_test = model.count_vectorizer_multinomial(test_texts, is_train=False)

    y_train = np.array(train_labels)
    model.train(X_train, y_train)
    predicted_labels, probas = model.predict(X_test)

    with open(args.out, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")

    print(probas)

    # Compare with the checker file
    data2 = np.load('./checker_files_v5/multinomial_bigrams_probas_test.npy')
    print("given")
    print(data2)
    test_predicted_indices = np.argmax(data2, axis=1)
    predicted_indices = np.argmax(probas, axis=1)
    matches = np.sum(predicted_indices == test_predicted_indices)
    total = len(predicted_indices)
    percentage_match = (matches / total) * 100
    print(f"The percentage of matches between your script's argmax and the test probabilities is: {percentage_match:.2f}%")
