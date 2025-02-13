import numpy as np
import pandas as pd
from typing import List
import re

class NGramBase:
    def __init__(self):
        """
        Initialize basic n-gram configuration.
        :param n: The order of the n-gram (e.g., 2 for bigram, 3 for trigram).
        :param lowercase: Whether to convert text to lowercase.
        :param remove_punctuation: Whether to remove punctuation from text.
        """
        self.current_config = {}
        
        # Internal dictionaries for storing counts
        self.ngram_counts = {}
        self.context_counts = {}

    def method_name(self) -> str:
        return f"Method Name: {self.current_config['method_name']}"

    def update_config(self, config) -> None:
        """
        Override the current configuration. You can use this method to update
        the config if required
        :param config: The new configuration.
        """
        self.current_config = config

    def preprocess(self, text: str) -> str:
        """
        Preprocess text before n-gram extraction.
        Perform lowercasing and punctuation removal if specified in config.
        """
        if self.current_config.get("lowercase", False):
            text = text.lower()
        if self.current_config.get("remove_punctuation", False):
            text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text. By default, split on whitespace.
        You can enhance this method to handle special cases if desired.
        """
        return text.strip().split()

    def fit(self, data: List[List[str]]) -> None:
        """
        Build n-gram counts from the list of tokenized sentences.
        :param data: The input data. Each sentence is a list of tokens.
        """
        n = self.current_config.get("n", 2)  # default to 2 for bigram
        self.ngram_counts = {}
        self.context_counts = {}

        # For each sentence, generate n-grams and update counts
        for sentence in data:
            # Optionally pad the sentence for n-gram boundary (e.g. for bigrams)
            # Using (n-1) start tokens if you want <s> markers, etc.
            for i in range(len(sentence) - n + 1):
                ngram = tuple(sentence[i : i + n])
                context = tuple(sentence[i : i + n - 1]) if n > 1 else ()

                self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
                self.context_counts[context] = self.context_counts.get(context, 0) + 1

    def prepare_data_for_fitting(self, data: List[str], use_fixed=False) -> List[List[str]]:
        """
        Prepare data for fitting.
        :param data: The input data.
        :return: The prepared data (list of token lists).
        """
        processed = []
        if not use_fixed:
            for text in data:
                processed_text = self.preprocess(text)
                tokenized = self.tokenize(processed_text)
                processed.append(tokenized)
        else:
            for text in data:
                processed_text = self.fixed_preprocess(text)
                tokenized = self.fixed_tokenize(processed_text)
                processed.append(tokenized)
        return processed

    def fixed_preprocess(self, text: str) -> str:
        """
        Removes punctuation and converts text to lowercase.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text by splitting at spaces.
        """
        return text.split()

    def perplexity(self, text: str) -> float:
        """
        Compute the perplexity of the model given the text.
        :param text: The input text.
        :return: The perplexity of the model.
        """
        tokens = self.tokenize(self.preprocess(text))
        n = self.current_config.get("n", 2)

        # Calculate log probability of each n-gram
        log_prob_sum = 0.0
        total_ngrams = 0
        
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            context = tuple(tokens[i : i + n - 1]) if n > 1 else ()
            ngram_count = self.ngram_counts.get(ngram, 0)
            context_count = self.context_counts.get(context, 0)

            # Basic MLE probability (no smoothing here)
            if context_count > 0 and ngram_count > 0:
                prob = ngram_count / context_count
            else:
                # Avoid log(0); you could apply smoothing instead
                prob = 1e-9  # dummy small probability

            log_prob_sum += np.log(prob)
            total_ngrams += 1

        # Perplexity = exp(- (1 / total_ngrams) * log_prob_sum)
        if total_ngrams == 0:
            return float('inf')
        perplexity_value = np.exp(-log_prob_sum / total_ngrams)
        return perplexity_value

if __name__ == "__main__":
    tester_ngram = NGramBase()
    # Example config
    tester_ngram.update_config({
        "method_name": "Basic NGram",
        "n": 2,  # bigram
        "lowercase": True,
        "remove_punctuation": True
    })

    sample_data = [
        "This, is a test sentence.",
        "Another test sentence, for demonstration!"
    ]

    # Prepare and fit
    prepared = tester_ngram.prepare_data_for_fitting(sample_data)
    tester_ngram.fit(prepared)

    # Check perplexity on a sample sentence
    test_sentence = "This is a test"
    print("Perplexity:", tester_ngram.perplexity(test_sentence))
