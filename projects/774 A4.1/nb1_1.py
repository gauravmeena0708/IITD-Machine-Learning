import numpy as np
import argparse
import pandas as pd
from nltk.stem import PorterStemmer

class BernoulliNBClassifier:
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
        return ' '.join(processed_tokens)

    def count_vectorizer_binary(self, corpus, is_train=True):
        """Generates a binary count vectorizer with handling for unseen words in test data."""
        tokenized_sentences = [sentence.split() for sentence in corpus]
        
        if is_train:
            # Build the vocabulary from training data
            self.vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
            unknown_index = len(self.vocab)  # Index for unknown feature
            num_features = len(self.vocab) + 1  # Add one for unknown feature
        else:
            # Use the existing vocabulary from training
            unknown_index = len(self.vocab)  # Index for unknown feature
            num_features = len(self.vocab) + 1  # Add one for unknown feature

        binary_matrix = np.zeros((len(corpus), num_features), dtype=int)

        for i, sentence in enumerate(tokenized_sentences):
            has_unknown = False  # Track if there's an unknown word in the sentence
            for word in sentence:
                if word in self.vocab:
                    j = self.vocab.index(word)
                    binary_matrix[i, j] = 1
                else:
                    has_unknown = True  # Mark that an unknown word was found
            if has_unknown:
                binary_matrix[i, unknown_index] = 1  # Set unknown feature to 1 if any unseen words

        return binary_matrix

    def calculate_class_priors(self, y_train):
        priors = np.zeros(len(self.class_order))
        total_samples = y_train.shape[0]

        for idx, c in enumerate(self.class_order):
            class_count = np.sum(y_train == c)
            priors[idx] = class_count / total_samples

        return priors

    def calculate_conditional_probs(self, X_train, y_train):
        num_classes = len(self.class_order)
        num_features = X_train.shape[1]
        conditional_probs = np.zeros((num_classes, num_features))

        for idx, c in enumerate(self.class_order):
            X_c = X_train[y_train == c]
            conditional_probs[idx, :] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)

        return conditional_probs

    def train(self, X_train, y_train):
        self.priors = self.calculate_class_priors(y_train)
        self.conditional_probs = self.calculate_conditional_probs(X_train, y_train)

    def eval(self, X_test):
        num_samples = X_test.shape[0]  
        num_classes = len(self.class_order)

        entry_matrix = np.zeros((num_samples, num_classes))

        for i, sample in enumerate(X_test):  # Loop over test samples
            for j, c in enumerate(self.class_order):
                prior = self.priors[j]
                likelihood_sum = 0.0  # Use sum for log probabilities
                for k in range(len(sample)):
                    if sample[k] == 1:
                        likelihood_sum += np.log(self.conditional_probs[j, k])
                    else:
                        likelihood_sum += np.log(1 - self.conditional_probs[j, k])

                entry_matrix[i, j] = prior * np.exp(likelihood_sum)  # Combine prior and likelihood
        return entry_matrix

    def predict(self, X_test):
        proba_predictions = self.eval(X_test)
        predicted_indices = np.argmax(proba_predictions, axis=1)
        predicted_labels = [self.class_order[i] for i in predicted_indices]
        return predicted_labels, proba_predictions

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stop", type=str, required=True)
    args = parser.parse_args()

    model = BernoulliNBClassifier(class_order=["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"])
    stopwords = model.load_stopwords(args.stop)

    train_texts, train_labels = pd.read_csv(args.train, sep='\t', header=None, quoting=3)[[2, 1]].values.T
    train_texts = [model.preprocess_text(text, stopwords) for text in train_texts]

    test_texts, _ = pd.read_csv(args.test, sep='\t', header=None, quoting=3)[[2, 1]].values.T
    test_texts = [model.preprocess_text(text, stopwords) for text in test_texts]

    X_train = model.count_vectorizer_binary(train_texts, is_train=True)
    X_test = model.count_vectorizer_binary(test_texts, is_train=False)

    y_train = np.array(train_labels)
    model.train(X_train, y_train)

    predicted_labels, probas = model.predict(X_test)

    # Write predictions to output file
    with open(args.out, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")

    # Compare with checker probabilities
    print(probas)
    data2 = np.load('./check5/bernoulli_probas_test.npy')
    print("given")
    print(data2)
    
    test_predicted_indices = np.argmax(data2, axis=1)
    predicted_indices = np.argmax(probas, axis=1)
    matches = np.sum(predicted_indices == test_predicted_indices)
    total = len(predicted_indices)
    percentage_match = (matches / total) * 100
    print(f"The percentage of matches between your script's argmax and the test probabilities is: {percentage_match:.2f}%")
