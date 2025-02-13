import nltk
import optuna
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import editdistance  # Third-party library for edit distance
import numpy as np
import os
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class NGramModel:
    def __init__(self, n: int, lowercase=True, remove_punctuation=True, smoothing="kneser_ney", gamma=0.1):
        self.n = n
        self.model = KneserNeyInterpolated(n)  # Or use MLE(n) for debugging
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.vocab = set()
        if smoothing == "kneser_ney":
            self.model = KneserNeyInterpolated(n)
        elif smoothing == "laplace":
            self.model = Laplace(n)
        elif smoothing == "lidstone":
            self.model = Lidstone(gamma, n)  # gamma is a small constant
        else:
            self.model = MLE(n)


    def preprocess(self, text: str, tag=False):
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        if not tag:
            return tokens
        else:
            return tokens, nltk.pos_tag(tokens)

    def train(self, text: str):
        tokens = self.preprocess(text)
        # Build vocab *before* training the model.
        for token in tokens:
          self.vocab.add(token)
        train_data, vocab = padded_everygram_pipeline(self.n, [tokens])
        self.model.fit(train_data, vocab)

    def train_ngram_models(self, text: str, n_values):
        models = []
        for n in n_values:
            tokenized_text = [word_tokenize(sent.lower()) for sent in sent_tokenize(text)]
            train_data, vocab = padded_everygram_pipeline(n, tokenized_text)
            model = KneserNeyInterpolated(n)
            model.fit(train_data, vocab)
            models.append(model)
        return models


    def tokenize(self, text: str):
        return self.preprocess(text)

    def check_vocab(self, word):
        return word in self.vocab


# Function to load text from a file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Function to load misspelling pairs from a file
def load_misspellings(file_path):
    """Loads misspelling pairs from a text file."""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ' && ' in line:
                correct, corrupted = line.strip().split(' && ')
                pairs.append((correct, corrupted))
    return pairs

# Function to build vocabulary from text
def build_vocabulary(text):
    """Builds a vocabulary (set of unique words) from the given text."""
    tokens = word_tokenize(text.lower())  # Lowercase for consistency
    return set(tokens)

# Function to train an n-gram language model
def train_ngram_model(text, n):
    tokenized_text = [word_tokenize(sent.lower()) for sent in sent_tokenize(text)]
    train_data, vocab = padded_everygram_pipeline(n, tokenized_text)
    model = KneserNeyInterpolated(n)
    model.fit(train_data, vocab)
    return model

PATH = '/content/'
if not os.path.exists(PATH + 'train1.txt'):
    PATH = './data/'

try:
    train_text1 = load_text(PATH + 'train1.txt')
    train_text2 = load_text(PATH + 'train2.txt')
    misspellings = load_misspellings(PATH + 'misspelling_public.txt')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

training_text = train_text1 + train_text2

# Build vocabulary and train 4-gram language model
model = NGramModel(2)
vocabulary = model.preprocess(train_text1 + train_text2)
lm_models = model.train_ngram_models(train_text1 + train_text2, [2, 3, 4])
