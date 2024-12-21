import heapq
import random
from itertools import chain, combinations, product

class Agent(object):
    def __init__(self, phoneme_table, vocabulary) -> None:
        """
        Your agent initialization goes here. You can also add code but don't remove the existing code.
        """
        self.phoneme_table = phoneme_table
        self.vocabulary = vocabulary
        self.best_state = None

    def asr_corrector(self, environment, beam_width=5, max_iterations=10):
        """
        Corrects the ASR output by applying Beam Search.

        Parameters:
        - environment: Object containing the initial state and cost function.
        - beam_width: Number of top candidates to keep at each step.
        - max_iterations: Maximum number of iterations to run the search.
        
        Updates environment.best_state with the best correction found.
        """
        # Initial setup
        print("started working")
        self.best_state = environment.init_state
        cost = environment.compute_cost(environment.init_state)
        print(self.best_state)
        
        sentence = environment.init_state
        replacement_map = {}
        cost_cache = {}  # Dictionary to cache the computed costs

        for k, v in self.phoneme_table.items():
            for i in v:
                if i not in replacement_map:
                    replacement_map[i] = [k]
                else:
                    replacement_map[i].append(k)

        # def find_substring_indexes(string, substring):
        #     indexes = []
        #     start = 0

        #     while True:
        #         index = string.find(substring, start)
        #         if index == -1:
        #             break
        #         indexes.append(index)
        #         start = index + 1

        #     return indexes

        # def apply_replacement(sentence, index, replacement, key_length):
        #     return sentence[:index] + replacement + sentence[index + key_length:]

        # def generate_neighbours(sentence, replacement_map):
        #     neighbors = []
        #     for key, replacements in replacement_map.items():
        #         indexes = find_substring_indexes(sentence, key)
        #         for index in indexes:
        #             for replacement in replacements:
        #                 new_sentence = apply_replacement(sentence, index, replacement, len(key))
        #                 neighbors.append(new_sentence)
        #     return neighbors

        def get_dict(sentence):
            word_dict = []
            words = sentence.split()
            for word in words:
                word_dict_with_weights = generate_all_possible_sentences(word, replacement_map)[:50]
                word_dict.append([x for x,_ in word_dict_with_weights])
            return word_dict

        def generate_all_possible_sentences(sentence, replacement_map):
            words = list(sentence)  # Assuming sentence is a string of letters, not words.

            # Create a list of lists where each sublist contains tuples (char, weight)
            replacement_options = []
            for letter in words:
                if letter in replacement_map:
                    # Include the original letter with weight 0 and its possible replacements with weight 1
                    replacement_options.append([(letter, 0)] + [(replacement, 1) for replacement in replacement_map[letter]])
                else:
                    replacement_options.append([(letter, 0)])  # No replacement, so weight is 0
            #print(replacement_options)
            # Generate all possible sentences by taking the Cartesian product of the replacement options
            all_possible_sentences_with_weights = []
            for option in product(*replacement_options):
                sentence = ''.join([char for char, _ in option])
                total_weight = sum([weight for _, weight in option])
                all_possible_sentences_with_weights.append((sentence, total_weight))

            # Sort by weight
            all_possible_sentences_with_weights.sort(key=lambda x: x[1])

            return all_possible_sentences_with_weights

        
        



        def get_cost(sentence):
            if sentence not in cost_cache:
                cost_cache[sentence] = environment.compute_cost(sentence)
            return cost_cache[sentence]


        def update_sentence_min_cost(sentence, word_dict, k, m):
            words = sentence.split()
            current_cost = get_cost(sentence)
            heap = [(current_cost, sentence)]
            for word_index in range(len(words)):
                if word_index < len(word_dict):
                    possible_replacements = word_dict[word_index]
                    if len(possible_replacements) > m:
                        selected_replacements = possible_replacements[:m]
                    else:
                        selected_replacements = possible_replacements

                    for new_word in selected_replacements:
                        new_sentence = words[:word_index] + [new_word] + words[word_index + 1:]
                        new_sentence_str = " ".join(new_sentence)
                        cost = get_cost(new_sentence_str)

                        if heap[0][0] > cost:
                            heap.clear()
                            heap = [(cost, new_sentence_str)]
                            # Update the best state if a better cost is found
                            if cost < get_cost(self.best_state):
                                self.best_state = new_sentence_str

                if heap and heap[0][0] < current_cost:
                    words = heap[0][1].split()
                    current_cost = heap[0][0]
                    # Update the best state if a better cost is found
                    if current_cost < get_cost(self.best_state):
                        self.best_state = " ".join(words)
            return " ".join(words)

        k = 1 
        m = 20 
        word_dict = get_dict(sentence)
        final_sentence = update_sentence_min_cost(sentence, word_dict, k, m)

        print(f"{get_cost(environment.init_state)}:->{get_cost(final_sentence)}")
        self.best_state = final_sentence
