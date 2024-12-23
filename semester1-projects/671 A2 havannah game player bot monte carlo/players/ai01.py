import time
import random
import numpy as np
from helper import *

class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Initialize the AIPlayer Agent

        Parameters:
        player_number (int): Current player number (1 or 2)
        timer: Timer object to fetch the remaining time for any player
        """
        self.player_number = player_number
        self.type = 'ai01'
        self.player_string = 'Player {}: ai01'.format(player_number)
        self.timer = timer
        self.opponent_number = 2 if player_number == 1 else 1

    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Decide the best move for the current player by evaluating all valid moves,
        focusing on both blocking the opponent and creating winning opportunities.
        """
        # Get the valid moves on the board (unoccupied positions)
        valid_actions = get_valid_actions(state)

        # First, check if there's a winning move for the AI
        for action in valid_actions:
            state[action[0], action[1]] = self.player_number
            win, _ = check_win(state, action, self.player_number)
            state[action[0], action[1]] = 0
            if win:
                return action  # Take the winning move immediately

        # Check if the opponent can win in the next move or the move after that, and block it
        best_actions = None
        best_score = float('-inf')
        for action in valid_actions:
            state[action[0], action[1]] = self.player_number

            # Simulate the opponent's response for each valid move
            opponent_moves = get_valid_actions(state)
            max_opponent_threat_score = 0

            for opponent_move in opponent_moves:
                state[opponent_move[0], opponent_move[1]] = self.opponent_number
                # Check if the opponent can win in the next move
                win, _ = check_win(state, opponent_move, self.opponent_number)
                if win:
                    max_opponent_threat_score = max(max_opponent_threat_score, 100)  # High score for opponent's winning move
                else:
                    # Now simulate the AI's move after the opponent's move
                    second_step_valid_actions = get_valid_actions(state)
                    for second_step_action in second_step_valid_actions:
                        state[second_step_action[0], second_step_action[1]] = self.player_number
                        # Re-evaluate the board for the AI after the second step
                        second_step_win, _ = check_win(state, second_step_action, self.player_number)
                        if second_step_win:
                            max_opponent_threat_score = max(max_opponent_threat_score, -50)  # Reward for AI setting up a win
                        state[second_step_action[0], second_step_action[1]] = 0

                state[opponent_move[0], opponent_move[1]] = 0  # Reset the opponent's move

            # Subtract the opponent's best response threat from the AI's evaluation
            current_score = self.evaluate_action(state, action) - max_opponent_threat_score
            state[action[0], action[1]] = 0  # Reset the AI's move
 
            if current_score > best_score:
                best_score = current_score
                best_actions = [action]
            elif current_score == best_score:
                best_actions.append(action)

        # Return the best action found
        return random.choice(best_actions)

    def evaluate_action(self, state, action):
        """
        Evaluate the current action based on strategic importance, including proximity
        to edges and corners, potential to connect or block a win condition, and 
        strategic connections on the board.
        """
        score = 0
        dim = state.shape[0]

        # Temporarily place the piece for evaluation
        state[action[0], action[1]] = self.player_number

        # Decrease score for proximity to edges and corners
        if get_edge(action, dim) != -1:
            score += 2  # Further reduced value for edges
        if get_corner(action, dim) != -1:
            score += 3  # Further reduced value for corners

        # Check for Fork, Bridge, and Ring potential
        win, structure = check_win(state, action, self.player_number)
        if win:
            if structure == "fork":
                score += 200  # High value for Fork
            elif structure == "bridge":
                score += 150  # High value for Bridge
            elif structure == "ring":
                score += 200  # High value for Ring

        # Increase score if the move connects two edges or is part of a strategic pattern
        if self.creates_strategic_connection(state, action, dim):
            score += 100  # Increased value for strategic connections

        # Reset the state back after evaluation
        state[action[0], action[1]] = 0
        
        return score

    def creates_strategic_connection(self, state, action, dim):
        """
        Determine if placing a piece at `action` will connect two different edges,
        or create a significant strategic advantage.
        """
        edges_connected = set()
        neighbors = get_neighbours(dim, action)
        for neighbor in neighbors:
            if state[neighbor[0], neighbor[1]] == self.player_number:
                edge = get_edge(neighbor, dim)
                if edge != -1:
                    edges_connected.add(edge)
        return len(edges_connected) > 1  # Return True if more than one edge is connected
