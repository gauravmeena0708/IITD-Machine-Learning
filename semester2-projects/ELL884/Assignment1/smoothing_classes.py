# smoothing_classes.py

from ngram import NGramBase
from config import *
from collections import defaultdict
import numpy as np
import math
from typing import List, Tuple

class NoSmoothing(NGramBase):
    def __init__(self):
        super(NoSmoothing, self).__init__()
        self.update_config(no_smoothing)

    def estimate_probability(self, ngram_tuple, nminus1_tuple):
        numerator = self.counts_ngram[ngram_tuple]
        denominator = self.counts_nminus1[nminus1_tuple]
        # raw MLE
        if denominator == 0:
            return 0.0
        return numerator / denominator

class AddK(NGramBase):
    def __init__(self):
        super(AddK, self).__init__()
        self.update_config(add_k)
        self.k = self.current_config.get('k', 1.0)  # default

    def estimate_probability(self, ngram_tuple, nminus1_tuple):
        numerator = self.counts_ngram[ngram_tuple] + self.k
        denominator = self.counts_nminus1[nminus1_tuple] + self.k * len(self.vocab)
        if denominator == 0:
            return 0.0
        return numerator / denominator

class StupidBackoff(NGramBase):
    def __init__(self):
        super(StupidBackoff, self).__init__()
        self.update_config(stupid_backoff)
        self.alpha = self.current_config.get('alpha', 0.4)


    def estimate_probability(self, ngram_tuple, nminus1_tuple):
        numerator = self.counts_ngram[ngram_tuple]
        denominator = self.counts_nminus1[nminus1_tuple]
        if numerator > 0 and denominator > 0:
            return numerator / denominator
        else:
            # fallback
            # the fallback is P(w_n) scaled by alpha
            # E.g. for bigram, take count of w_n in unigrams / total
            wn = ngram_tuple[-1]
            wn_count = 0
            # we can do something like self.counts_nminus1[(wn,)] if n=2
            wn_count = self.counts_nminus1[(wn,)]
            if self.total_words == 0:
                return 1e-9
            return self.alpha * (wn_count / self.total_words)

class GoodTuring(NGramBase):
    def __init__(self):
        super(GoodTuring, self).__init__()
        self.update_config(good_turing)
        # place to store counts-of-counts, etc.
        self.max_seen_count = 0
        self.freq_of_freq = defaultdict(int)

    def fit(self, data):
        super(GoodTuring, self).fit(data)
        # after building ngram counts, build frequency-of-frequency table
        # and max_seen_count
        for c in self.counts_ngram.values():
            self.freq_of_freq[c] += 1
            if c > self.max_seen_count:
                self.max_seen_count = c

    def estimate_probability(self, ngram_tuple, nminus1_tuple):
        c = self.counts_ngram[ngram_tuple]
        c_next = c+1
        if c_next <= self.max_seen_count and self.freq_of_freq[c] != 0:
            # Good-Turing discount
            numerator = (c_next * self.freq_of_freq[c_next]) / self.freq_of_freq[c]
        else:
            # fallback if c is large or not found
            numerator = c
        denominator = sum([count for count in self.counts_ngram.values()])  # sum of all bigram/trigram counts
        return numerator / denominator if denominator != 0 else 1e-9

class Interpolation(NGramBase):
    def __init__(self):
        super(Interpolation, self).__init__()
        self.update_config(interpolation)
        # lambdas is a list, e.g. for trigram [lambda1, lambda2, lambda3].
        # We expect sum(lambdas)=1
        self.lambdas = self.current_config.get('lambdas', [0.4, 0.3, 0.3])

    def estimate_probability(self, ngram_tuple, nminus1_tuple):
        """
        Example for trigram interpolation.
        P_interpolated = λ1 * p(w_n|w_{n-1},w_{n-2}) + λ2 * p(w_n|w_{n-1}) + λ3 * p(w_n)
        If self.n=3, ngram_tuple = (w_{n-2}, w_{n-1}, w_n).
        """
        # handle trigram, bigram, unigram counts
        w_n   = ngram_tuple[-1]
        bigram_tuple = (ngram_tuple[-2], w_n)  # or for n>2
        unigram_tuple = (w_n,)

        trigram_count = self.counts_ngram[ngram_tuple]
        bigram_count  = self.counts_ngram[bigram_tuple]
        unigram_count = self.counts_nminus1[unigram_tuple]  # same as counts of w_n ?

        two_of_three = (ngram_tuple[0], ngram_tuple[1])  # w_{n-2}, w_{n-1}
        # For the trigram's (nminus1), if the n=3.
        trigram_den = self.counts_nminus1[two_of_three]

        one_of_three = (ngram_tuple[-2],)  # w_{n-1}
        bigram_den   = self.counts_nminus1[one_of_three]

        # total words for unigrams
        # or sum up all unigrams
        denom_unigram = sum(self.counts_nminus1.values())

        p_trigram = trigram_count / trigram_den if trigram_den > 0 else 0
        p_bigram  = bigram_count / bigram_den if bigram_den > 0 else 0
        p_unigram = unigram_count / denom_unigram if denom_unigram > 0 else 1e-9

        λ1, λ2, λ3 = self.lambdas
        return λ1*p_trigram + λ2*p_bigram + λ3*p_unigram

class KneserNey(NGramBase):
    def __init__(self):
        super(KneserNey, self).__init__()
        self.update_config(kneser_ney)
        self.discount = self.current_config.get('discount', 0.75)

    def estimate_probability(self, ngram_tuple, nminus1_tuple):
        # discount
        d = self.discount
        bigram_count = self.counts_ngram[ngram_tuple]
        w_prev_count = self.counts_nminus1[nminus1_tuple]

        # how many distinct words appear after w_{n-1}?
        # naive approach: iterate once. For large corpora, you'd precompute in fit().
        if w_prev_count == 0:
            return 1e-9

        # numerator
        adjusted = max(bigram_count - d, 0) / w_prev_count

        # For continuation probability, do a quick pass
        # # distinct words that appear after w_{n-1}...
        # again, you likely want to precompute these, but let's do naive for demonstration:
        distinct_after_wprev = 0
        for (bg), ct in self.counts_ngram.items():
            if bg[0] == nminus1_tuple[0]:
                distinct_after_wprev += 1

        # # distinct bigrams that end with w_n ...
        ends_with_wn = 0
        for (bg), ct in self.counts_ngram.items():
            if bg[1] == ngram_tuple[1]:  # w_n
                ends_with_wn += 1
        total_bigrams = len(self.counts_ngram)

        lam = d * distinct_after_wprev / w_prev_count
        p_cont = ends_with_wn / total_bigrams

        return adjusted + lam * p_cont
class GoodTuring(NGramBase):
    def __init__(self):
        super(GoodTuring, self).__init__()
        self.update_config(good_turing)
        self.c_star = None
        self.N = None
        self.frequency_counts = None

    def calculate_good_turing(self):
        ngram_counts = self.unigram_counts if self.current_config['n'] == 1 else self.ngram_counts
        self.frequency_counts = defaultdict(int)
        
        for count in ngram_counts.values():
            self.frequency_counts[count] += 1

        counts = sorted(self.frequency_counts.keys())
        self.N = sum(count * self.frequency_counts[count] for count in counts)
        self.c_star = {}
        
        # Simple Good-Turing implementation
        k = 5  # Threshold for switching to linear regression
        for c in counts:
            if c < k and (c+1) in self.frequency_counts:
                self.c_star[c] = (c + 1) * self.frequency_counts[c + 1] / self.frequency_counts[c]
            else:
                # Linear regression for high frequencies
                log_r = np.log([r for r in counts if r >= k])
                log_Nr = np.log([self.frequency_counts[r] for r in counts if r >= k])
                slope, intercept, _, _, _ = linregress(log_r, log_Nr)
                self.c_star[c] = np.exp(intercept + slope * np.log(c))

        # Handle unseen events (N_1/N normalized by vocabulary)
        vocab_size = len(ngram_counts)
        total_possible = (vocab_size ** self.current_config['n']) - vocab_size
        self.c_star[0] = self.frequency_counts[1] / (self.N * total_possible) if total_possible > 0 else 0

    def sentence_probability(self, sentence: List[str]) -> float:
        probability = 1.0
        n = self.current_config['n']
        padded = [''] * (n-1) + sentence
        
        for i in range(n-1, len(padded)):
            ngram = tuple(padded[i-n+1:i+1])
            count = self.unigram_counts[ngram[0]] if n == 1 else self.ngram_counts.get(ngram, 0)
            probability *= self.c_star.get(count, self.c_star[0])
        
        return probability

class KneserNey(NGramBase):
    def calculate_kneser_ney_counts(self, data: List[List[str]]):
        n = self.current_config['n']
        self.lower_order_continuations = defaultdict(lambda: defaultdict(int))

        # Calculate continuation counts for all orders
        for order in range(1, n+1):
            for sentence in data:
                padded = [''] * (order-1) + sentence + ['']
                for i in range(order-1, len(padded)):
                    context = tuple(padded[i-order+1:i])
                    word = padded[i]
                    
                    if order == 1:
                        # For unigrams: count distinct left contexts
                        if i > 0:
                            left_context = padded[i-1]
                            self.lower_order_continuations[1][word] += 1
                    else:
                        # For higher orders: count distinct contexts
                        self.lower_order_continuations[order][context] += 1

    def sentence_probability(self, sentence: List[str]) -> float:
        n = self.current_config['n']
        discount = self.current_config.get('discount', 0.75)
        probability = 1.0
        padded = [''] * (n-1) + sentence + ['']

        for i in range(n-1, len(padded)-1):
            ngram = tuple(padded[i-n+1:i+1])
            context = tuple(padded[i-n+1:i])
            count = self.ngram_counts.get(ngram, 0)
            context_count = self.ngram_counts.get(context, 0) if n > 1 else self.N
            
            # Calculate discounted probability
            if context_count > 0:
                prob = max(count - discount, 0) / context_count
            else:
                prob = 0

            # Calculate lambda factor
            unique_followers = len([k for k in self.ngram_counts if k[:-1] == context])
            lambda_ = (discount * unique_followers) / context_count if context_count > 0 else 1

            # Calculate lower-order probability
            if n > 1:
                lower_order_prob = self.lower_order_continuations[n].get(ngram[1:], 0) / self.N
            else:
                lower_order_prob = self.lower_order_continuations[1].get(ngram[0], 0) / self.N

            probability *= (prob + lambda_ * lower_order_prob)

        return probability

class Interpolation(NGramBase):
    def fit(self, data: List[List[str]]) -> None:
        n = self.current_config['n']
        super().fit(data)
        self.ngram_models = []

        # Create recursive interpolation models
        for i in range(n-1, 0, -1):
            model = Interpolation() if i > 1 else AddK()
            model.update_config({'n': i, 'lambdas': self.current_config['lambdas'][-i:]})
            model.fit(data)
            self.ngram_models.append(model)

    def sentence_probability(self, sentence: List[str]) -> float:
        n = self.current_config['n']
        lambdas = self.current_config['lambdas'][:n]
        total = sum(lambdas)
        lambdas = [lam/total for lam in lambdas]
        probability = 1.0

        for i in range(len(sentence)-n+1):
            ngram = tuple(sentence[i:i+n])
            weighted_prob = 0
            
            for j in range(n):
                context = sentence[i+j:i+n]
                if len(context) == 0:
                    continue
                weighted_prob += lambdas[j] * self.ngram_models[j].sentence_probability(context)
            
            probability *= weighted_prob

        return probability
