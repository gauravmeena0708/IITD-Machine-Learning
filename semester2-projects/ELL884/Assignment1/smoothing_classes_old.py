from ngram import NGramBase
from config import *
import numpy as np
import pandas as pd

#adding basic and common imports
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import math
import os, re
import math

class NoSmoothing(NGramBase):
    def __init__(self, n=2):

        super().__init__(n=n)
        self.update_config(no_smoothing)
        self.vocab = set()

    def prob(self, ngram: tuple) -> float:
        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return 0.0
        return self.ngram_counts.get(ngram, 0) / context_count


class AddK(NGramBase):
    def __init__(self, n=2):
        super().__init__(n=n)
        self.update_config(add_k)
        self.k = self.current_config.get('k', 1.0)

    def prob(self, ngram: tuple) -> float:
        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)
        ngram_count = self.ngram_counts.get(ngram, 0)
        V = len(self.vocabulary)

        return (ngram_count + self.k) / (context_count + self.k * V) if context_count > 0 else (self.k / (self.k * V))


class StupidBackoff(NGramBase):
    def __init__(self, n=2):
        super().__init__(n=n)
        self.update_config(stupid_backoff)
        self.alpha = self.current_config.get('alpha', 0.4)

    def prob(self, ngram: tuple) -> float:
        context = ngram[:-1]
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        if ngram_count > 0 and context_count > 0:
            return ngram_count / context_count
        else:
            word = (ngram[-1],)
            unigram_count = self.context_counts.get(word, 0)  
            total_unigrams = sum(self.context_counts[c] for c in self.context_counts if len(c) == 1)
            p_unigram = (unigram_count / total_unigrams) if total_unigrams > 0 else 0.0
            return self.alpha * p_unigram


class GoodTuring(NGramBase):
    def __init__(self, n=2):
        super().__init__(n=n)
        self.update_config(good_turing)
        self.freq_of_freq = defaultdict(int)  
        self.good_turing_map = {}

    def fit(self, data):
        super().fit(data)


        for count_val in self.ngram_counts.values():
            self.freq_of_freq[count_val] += 1

        max_count = max(self.ngram_counts.values()) if self.ngram_counts else 0
        for c in range(max_count + 1):
            Nc = self.freq_of_freq.get(c, 0)
            Nc1 = self.freq_of_freq.get(c+1, 0)
            if Nc > 0 and Nc1 > 0:
                self.good_turing_map[c] = (c + 1) * (Nc1 / Nc)
            else:
                self.good_turing_map[c] = c

    def prob(self, ngram: tuple) -> float:
        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return 0.0
        
        c = self.ngram_counts.get(ngram, 0)
        c_star = self.good_turing_map.get(c, c)
        return c_star / context_count


class Interpolation(NGramBase):
    def __init__(self, n=2):
        super().__init__(n=n)
        self.update_config(interpolation)
        self.lambdas = self.current_config.get('lambdas', [0.5, 0.5])

    def prob(self, ngram: tuple) -> float:
        if len(self.lambdas) != 2:
            raise ValueError("Interpolation expects two lambdas for bigram+unigram.")

        lam_bigram = self.lambdas[0]
        lam_unigram = self.lambdas[1]
        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)
        ngram_count = self.ngram_counts.get(ngram, 0)


        p_bigram = 0.0
        if context_count > 0:
            p_bigram = ngram_count / context_count

        word = (ngram[-1],)
        word_count = self.context_counts.get(word, 0)
        total_unigrams = sum(self.context_counts[c] for c in self.context_counts if len(c) == 1)
        p_unigram = (word_count / total_unigrams) if total_unigrams > 0 else 0.0

        return lam_bigram * p_bigram + lam_unigram * p_unigram


class KneserNey(NGramBase):
    def __init__(self, n=2):
        super().__init__(n=n)
        self.update_config(kneser_ney)
        self.discount = self.current_config.get('discount', 0.75)

    def prob(self, ngram: tuple) -> float:
        context = ngram[:-1]
        bigram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return self._continuation_prob(ngram[-1])

        numerator = max(bigram_count - self.discount, 0)
        discounted_mle = numerator / context_count


        distinct_followers_of_context = self._distinct_followers(context)
        lambda_context = (self.discount * distinct_followers_of_context) / context_count
        return discounted_mle + lambda_context * self._continuation_prob(ngram[-1])

    def _distinct_followers(self, context: tuple) -> int:
        distinct_set = set()
        for bg in self.ngram_counts:
            if bg[:-1] == context:
                distinct_set.add(bg[-1])
        return len(distinct_set)

    def _continuation_prob(self, word: str) -> float:
        distinct_left_contexts = 0
        total_bigram_types = 0
        for bg in self.ngram_counts:
            if len(bg) == 2:
                total_bigram_types += 1
                if bg[-1] == word:
                    distinct_left_contexts += 1
        
        if total_bigram_types == 0:
            return 0.0
        return distinct_left_contexts / total_bigram_types



if __name__ == "__main__":
    def load_text(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_misspellings(file_path):
        """Loads misspelling pairs from a text file."""
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ' && ' in line:
                    correct, corrupted = line.strip().split(' && ')
                    pairs.append((correct, corrupted))
        return pairs

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
    vocabulary = train_text1 + "\n" + train_text2
    tester_ngram = NGramBase(n=2)
    lines = tester_ngram.preprocess(vocabulary).split('\n')
    tokenized_data = tester_ngram.prepare_data_for_fitting(lines)
    padded_data = tester_ngram.add_padding(tokenized_data)
   
    # --- No Smoothing ---
    ns = NoSmoothing()
    ns.fit(padded_data)
    
    # --- Add-k Smoothing ---
    addk = AddK()
    addk.fit(padded_data)
    
    # --- Stupid Backoff ---
    sb = StupidBackoff()
    sb.fit(padded_data)
    
    # --- Good-Turing ---
    gt = GoodTuring()
    gt.fit(padded_data)

    # --- Interpolation ---
    interp = Interpolation()
    interp.fit(padded_data)

    # --- Kneser-Ney ---
    kn = KneserNey()
    kn.fit(padded_data)

    print("-"*70)
    print()
    print("Probablities correct")
    tokens = ('this', 'is')
    print(f"No Smoothing: {ns.prob(tokens)}")  
    print(f"Add-k: {addk.prob(tokens)}")  
    print(f"Stupid Backoff (alpha=0.4): {sb.prob(tokens)}")  
    print(f"Good-Turing: {gt.prob(tokens)}") 
    print(f"Interpolation : {interp.prob(tokens)}")  
    print(f"Kneser-Ney: {kn.prob(tokens)}")  

    print("-"*70)
    print()
    print("Probablities corrupt")
    tokens = ('this', 'sdhdhf')
    print(f"No Smoothing: {ns.prob(tokens)}")  
    print(f"Add-k: {addk.prob(tokens)}")  
    print(f"Stupid Backoff (alpha=0.4): {sb.prob(tokens)}")  
    print(f"Good-Turing: {gt.prob(tokens)}") 
    print(f"Interpolation : {interp.prob(tokens)}")  
    print(f"Kneser-Ney: {kn.prob(tokens)}") 

    print("-"*70)
    print()
    print("Perplexities correct")
    sentense = 'this is a test'
    print(f"No Smoothing Perplexity: {ns.perplexity(sentense)}")
    print(f"Add-k Perplexity: {addk.perplexity(sentense)}")
    print(f"Stupid Backoff Perplexity: {sb.perplexity(sentense)}")
    print(f"Good-Turing Perplexity: {gt.perplexity(sentense)}")
    print(f"Interpolation Perplexity: {interp.perplexity(sentense)}")
    print(f"Kneser-Ney Perplexity: {kn.perplexity(sentense)}")

    print("-"*70)
    print()
    print("Perplexities incorrect")
    sentense = 'find aizceq asdg tst'
    print(f"No Smoothing Perplexity: {ns.perplexity(sentense)}")
    print(f"Add-k Perplexity: {addk.perplexity(sentense)}")
    print(f"Stupid Backoff Perplexity: {sb.perplexity(sentense)}")
    print(f"Good-Turing Perplexity: {gt.perplexity(sentense)}")
    print(f"Interpolation Perplexity: {interp.perplexity(sentense)}")
    print(f"Kneser-Ney Perplexity: {kn.perplexity(sentense)}")
    
 
