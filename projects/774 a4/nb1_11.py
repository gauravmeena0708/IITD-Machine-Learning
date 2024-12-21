import numpy as np
import argparse
import pandas as pd
from nltk.stem import PorterStemmer
import pandas as pd

def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        stopwords = set(line.strip() for line in file)
    return stopwords

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None)#, quoting=3)
    texts = data[2].values 
    labels = data[1].values  
    return texts, labels

def preprocess_text(text, stop_words):
    ps = PorterStemmer()
    tokens = text.lower().split() 
    processed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)


def count_vectorizer_binary(corpus, vocab=None):
    tokenized_sentences = [sentence.split() for sentence in corpus]
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    binary_matrix = np.zeros((len(corpus), len(vocab)), dtype=int)
    
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in vocab:
                j = vocab.index(word)
                binary_matrix[i, j] = 1 
    
    return binary_matrix, vocab


def calculate_class_priors(y_train, classes):
    priors = np.zeros(len(classes))
    total_samples = y_train.shape[0]
    
    for idx, c in enumerate(classes):
        class_count = np.sum(y_train == c)
        priors[idx] = class_count / total_samples
    
    return priors

def calculate_conditional_probs(X_train, y_train, classes):
    num_classes = len(classes)
    num_features = X_train.shape[1]
    conditional_probs = np.zeros((num_classes, num_features))
    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        conditional_probs[idx, :] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)
    return conditional_probs


def predict_class(entry_matrix):
    predicted_indices = np.argmax(entry_matrix, axis=1)
    return predicted_indices


def bernoulli_nb(X_train, y_train, X_test, classes):
    priors = calculate_class_priors(y_train, classes)
    conditional_probs = calculate_conditional_probs(X_train, y_train, classes)

    num_samples = X_test.shape[0]  
    num_classes = len(classes)
    
    entry_matrix = np.zeros((num_samples, num_classes))
    
    for i, sample in enumerate(X_test):  # Loop over test samples
        for j, c in enumerate(classes):
            prior = priors[j]
            likelihood_product = 1.0  
            for k in range(len(sample)):
                if sample[k] == 1:
                    likelihood_product *= conditional_probs[j, k]  
                # else:
                #     likelihood_product *= (1 - conditional_probs[j, k])  
            entry_matrix[i, j] = prior * likelihood_product
    return entry_matrix


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stop", type=str, required=True)
    args = parser.parse_args()

    stopwords = load_stopwords(args.stop)
    train_texts, train_labels = load_data(args.train)
    print(f"Number of samples in training data: {len(train_texts)}")
    train_texts = [preprocess_text(text, stopwords) for text in train_texts]
    test_texts, _ = load_data(args.test)
    test_texts = [preprocess_text(text, stopwords) for text in test_texts]
    
    class_order = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

    X_train, vocab = count_vectorizer_binary(train_texts)
    X_test, _ = count_vectorizer_binary(test_texts, vocab)

    y_train = np.array(train_labels)
    proba_predictions = bernoulli_nb(X_train, y_train, X_train, class_order)
    print(proba_predictions)
    predicted_indices = np.argmax(proba_predictions, axis=1)
    predicted_labels = [class_order[i] for i in predicted_indices]

    with open(args.out, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")

    data2 = np.load('./check/bernoulli_probas_train.npy')
    print(f"Shape of loaded probabilities from bernoulli_probas_train.npy: {data2.shape}")
    print("given:")
    print(data2)
    test_predicted_indices = np.argmax(data2, axis=1)
    matches = np.sum(predicted_indices == test_predicted_indices)
    total = len(predicted_indices)
    percentage_match = (matches / total) * 100
    print(f"The percentage of matches between your script's argmax and the test probabilities is: {percentage_match:.2f}%")
