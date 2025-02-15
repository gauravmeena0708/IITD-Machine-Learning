import numpy as np
import pandas as pd
from typing import List, Tuple
import itertools
import editdistance

from smoothing_classes import *
from config import error_correction

class SpellingCorrector:
    def __init__(self, dictionary_path="./data/words_list.txt", method="ADD_K", min_threshold=0.01, edit_factor=0.01):
        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']

        if self.internal_ngram_name == "NO_SMOOTH":
            self.internal_ngram = NoSmoothing()
        elif self.internal_ngram_name == "ADD_K":
            self.internal_ngram = AddK()
        elif self.internal_ngram_name == "STUPID_BACKOFF":
            self.internal_ngram = StupidBackoff()
        elif self.internal_ngram_name == "GOOD_TURING":
            self.internal_ngram = GoodTuring()
        elif self.internal_ngram_name == "INTERPOLATION":
            self.internal_ngram = Interpolation()
        elif self.internal_ngram_name == "KNESER_NEY":
            self.internal_ngram = KneserNey()

        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])
        self.correction_config = error_correction
        self.method = method
        self.internal_ngram = self.select_ngram_model()
        self.dictionary = self._load_dictionary(dictionary_path)
        self.ngram_cache = {}

        self.min_threshold = min_threshold
        self.edit_factor = edit_factor  # Corrected attribute name

    def select_ngram_model(self):
        models = {
            "NO_SMOOTH": NoSmoothing,
            "ADD_K": AddK,
            "STUPID_BACKOFF": StupidBackoff,
            "GOOD_TURING": GoodTuring,
            "INTERPOLATION": Interpolation,
            "KNESER_NEY": KneserNey
        }
        # method = method  # self.correction_config['internal_ngram_best_config']['method_name']
        logging.debug(f"Selected n-gram smoothing method: {self.method}")
        return models.get(self.method, NoSmoothing)()

    def fit(self, data: List[str]) -> None:
        """
        Fit the n-gram language model on the provided training data.
        """
        tokenized_data = self.internal_ngram.prepare_data_for_fitting(data)
        padded_data = self.internal_ngram.add_padding(tokenized_data)
        self.internal_ngram.fit(padded_data)

    def correct(self, texts: List[str]) -> List[str]:
        """
        Corrects spelling for each line in the input list of texts.
        Returns the corrected lines.
        """
        corrected_texts = []
        for line_idx, line in enumerate(texts):
            tokens = line.strip().split()
            corrected_line = []
            for idx, word in enumerate(tokens):
                # Generate candidate corrections
                candidates = self.generate_candidates(word, top_n=10)
                # Pick the best candidate using the n-gram model + edit distance
                best_candidate = self.pick_best_candidate(tokens, idx, candidates)
                corrected_line.append(best_candidate)
            corrected_texts.append(" ".join(corrected_line))

            logging.debug(f"Original line {line_idx}: {line}")
            logging.debug(f"Corrected line {line_idx}: {corrected_texts[-1]}")
            logging.debug(f"Perplexity line {line_idx}: {self.internal_ngram.perplexity(corrected_texts[-1])}")
        return corrected_texts

    def generate_candidates(self, word: str, top_n=10) -> List[str]:
        """
        Generates candidate corrections by:
          1. Sorting vocabulary words by edit distance (ascending).
          2. Providing some top-N candidates.
          3. Checking if the word can be split into two valid dictionary words.

        Returns a list of candidate words (strings).
        """
        # 1. Sort by edit distance
        sorted_by_distance = sorted(
            ((vocab_word, editdistance.eval(word, vocab_word))
             for vocab_word in self.internal_ngram.vocab),
            key=lambda x: x[1]
        )

        # Keep top_n candidates
        candidates = [w[0] for w in sorted_by_distance[:top_n]]
        splits = self._generate_word_splits(word)
        for split_word in splits:
            if split_word not in candidates:
                candidates.append(split_word)

        return candidates

    def pick_best_candidate(self, tokens: List[str], idx: int, candidates: List[str]) -> str:
        if not candidates:
            return tokens[idx]  # Fallback to original if no candidates

        original_word = tokens[idx]
        best_candidate = original_word
        best_score = float('-inf')

        for c in candidates:
            lm_prob = self._compute_ngram_probability(tokens, idx, c)
            score = self._compute_candidate_score(original_word, c, lm_prob)

            if score > best_score:
                best_score = score
                best_candidate = c

        if best_score < self.min_threshold:
            logging.debug(
                f"Best candidate '{best_candidate}' for word '{original_word}' "
                f"has low score ({best_score:.5f}). Keeping original."
            )
            return original_word

        return best_candidate

    def _load_dictionary(self, dictionary_path: str) -> set:
        """
        Loads words from a dictionary file (one word per line).
        Returns a set of valid words (lowercased).
        """
        try:
            with open(dictionary_path, "r", encoding="utf-8") as f:
                words = {line.strip().lower() for line in f if line.strip()}
            logging.info(f"Dictionary loaded from {dictionary_path}, total words: {len(words)}")
            return words
        except FileNotFoundError:
            logging.error(f"Dictionary file not found: {dictionary_path}")
            raise

    ############################################################################
    # EXTRA HELPER METHODS BELOW (DO NOT CHANGE FUNCTION SIGNATURES ABOVE)     #
    ############################################################################

    def _generate_word_splits(self, word: str) -> List[str]:
        splits = []
        for i in range(1, len(word)):
            part1, part2 = word[:i], word[i:]
            if part1 in self.dictionary and part2 in self.dictionary:
                splits.append(f"{part1} {part2}")
        return splits

    def _compute_ngram_probability(self, tokens: List[str], idx: int, candidate: str) -> float:
        prev_word = tokens[idx - 1] if idx - 1 >= 0 else '<s>'
        ngram_tuple = (prev_word, candidate)
        nminus1_tuple = (prev_word,)

        if ngram_tuple in self.ngram_cache:
            return self.ngram_cache[ngram_tuple]

        lm_prob = self.internal_ngram.prob(ngram_tuple, nminus1_tuple)
        self.ngram_cache[ngram_tuple] = lm_prob
        return lm_prob

    def _compute_candidate_score(self, original_word: str, candidate: str, lm_prob: float) -> float:
        edit_dist = editdistance.eval(original_word, candidate)
        return np.log(lm_prob + 1e-9) - self.edit_factor * edit_dist  # Use corrected attribute


if __name__ == "__main__":
    def load_training_datasets(paths=['./data/train1.txt', './data/train2.txt']) -> List[str]:
        raw_training_texts = []
        for file in paths:
            with open(file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                raw_training_texts.append(line)
        return raw_training_texts

    def load_test_dataset(path: str = './data/misspelling_public.txt') -> Tuple[List[str], List[str]]:
        corrupt, truth = [], []
        with open(path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            try:
                t, c = line.strip().split('&&')
                corrupt.append(c)
                truth.append(t)
            except:
                pass
        return corrupt, truth


    METHOD_NAME = ["NO_SMOOTH", "ADD_K", "STUPID_BACKOFF", "GOOD_TURING", "INTERPOLATION", "KNESER_NEY"]
    MIN_THRESHOLD = 0.01
    EDITDIST_FACTOR = 0.01
    corrector = SpellingCorrector(method=METHOD_NAME[0], min_threshold=MIN_THRESHOLD, edit_factor=EDITDIST_FACTOR)
    raw_training_texts = load_training_datasets()
    corrupt, truth = load_test_dataset()
    corrector.fit(raw_training_texts)
    from scoring import score_batch
    print("Vocab size:", len(corrector.internal_ngram.vocab))
    corrected_sentences = corrector.correct(corrupt)
    sc = score_batch(corrected_sentences, truth)
    print(sc)