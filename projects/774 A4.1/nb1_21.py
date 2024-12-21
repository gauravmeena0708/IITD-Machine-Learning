import numpy as np
import argparse
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
        return ' '.join(processed_tokens)

    def count_vectorizer_multinomial(self, corpus):
        tokenized_sentences = [sentence.split() for sentence in corpus]
        if self.vocab is None:
            self.vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))

        count_matrix = np.zeros((len(corpus), len(self.vocab)), dtype=int)

        for i, sentence in enumerate(tokenized_sentences):
            for word in sentence:
                if word in self.vocab:
                    j = self.vocab.index(word)
                    count_matrix[i, j] += 1 

        return count_matrix

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
            conditional_probs[idx, :] = (np.sum(X_c, axis=0) + 1) / (np.sum(X_c) + num_features)

        return conditional_probs

    def train(self, X_train, y_train):
        self.priors = self.calculate_class_priors(y_train)
        self.conditional_probs = self.calculate_conditional_probs(X_train, y_train)

    def eval(self, X_test):
        num_samples = X_test.shape[0]
        num_classes = len(self.class_order)

        entry_matrix = np.zeros((num_samples, num_classes))

        for i, sample in enumerate(X_test): 
            for j, c in enumerate(self.class_order):
                prior = self.priors[j]
                likelihood_product = prior 
                for k in range(len(sample)):
                    likelihood_product *= self.conditional_probs[j, k] ** sample[k] 
                entry_matrix[i, j] = likelihood_product

        return entry_matrix

    def predict(self, X_test):
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

    # Initialize model
    model = MultinomialNBClassifier(class_order=["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"])
    stopwords = model.load_stopwords(args.stop)
    train_texts, train_labels = pd.read_csv(args.train, sep='\t', header=None)[[2, 1]].values.T
    train_texts = [model.preprocess_text(text, stopwords) for text in train_texts]
    test_texts, _ = pd.read_csv(args.test, sep='\t', header=None)[[2, 1]].values.T
    test_texts = [model.preprocess_text(text, stopwords) for text in test_texts]
    X_train = model.count_vectorizer_multinomial(train_texts)
    X_test = model.count_vectorizer_multinomial(test_texts)

    #Train
    y_train = np.array(train_labels)
    model.train(X_train, y_train)

    #Test
    predicted_labels, probas = model.predict(X_test)
    print(probas)
    data2 = np.load('./check/multinomial_probas_test.npy')
    print("given")
    print(data2)
    with open(args.out, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")
    test_predicted_indices = np.argmax(data2, axis=1)
    predicted_indices = np.argmax(probas, axis=1)
    matches = np.sum(predicted_indices == test_predicted_indices)
    total = len(predicted_indices)
    percentage_match = (matches / total) * 100
    print(f"The percentage of matches between your script's argmax and the test probabilities is: {percentage_match:.2f}%")
