import time
import math
import random
import numpy as np
from helper import *


class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = 'ai10'
        self.player_string = 'Player {}: ai10'.format(player_number)
        self.timer = timer
        self.opponent_number = 2 if player_number == 1 else 1


    # def get_move(self, state: np.array) -> Tuple[int, int]:
    #     """
    #     Decide the best move for the current player by evaluating all valid moves
    #     and selecting the move that maximizes the player's chances of winning while
    #     preventing the opponent from winning in the next move.
    #     """
    #     # Get the board size (assuming it's square)
    #     dim = state.shape[0]
        
    #     # Get the valid moves on the board (unoccupied positions)
    #     valid_actions = self.get_valid_actions(state)
        
    #     # First, check if there's a winning move for the AI
    #     for action in valid_actions:
    #         state[action[0], action[1]] = self.player_number
    #         win, _ = check_win(state, action, self.player_number)
    #         state[action[0], action[1]] = 0
    #         if win:
    #             return action  # Take the winning move immediately

    #     # Next, check if the opponent can win in the next move and block it
    #     for action in valid_actions:
    #         state[action[0], action[1]] = self.opponent_number
    #         win, _ = check_win(state, action, self.opponent_number)
    #         state[action[0], action[1]] = 0
    #         if win:
    #             return action  # Block the opponent's winning move

    #     # If no immediate wins or threats, evaluate the best move
    #     best_action = None
    #     best_score = float('-inf')
    #     for action in valid_actions:
    #         state[action[0], action[1]] = self.player_number
    #         current_score = self.evaluate_action(state, action, self.get_neighbours(dim, action), dim)
    #         state[action[0], action[1]] = 0
            
    #         if current_score > best_score:
    #             best_action = action
    #             best_score = current_score
        
    #     # Return the best action found
    #     return best_action
    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Decide the best move for the current player by evaluating all valid moves
        and selecting the move that maximizes the player's chances of winning while
        preventing the opponent from winning in the next move or in two sequential moves.
        """
        # Get the board size (assuming it's square)
        dim = state.shape[0]
        
        # Get the valid moves on the board (unoccupied positions)
        valid_actions = self.get_valid_actions(state)
        
        # First, check if there's a winning move for the AI
        for action in valid_actions:
            state[action[0], action[1]] = self.player_number
            win, _ = check_win(state, action, self.player_number)
            state[action[0], action[1]] = 0
            if win:
                return action  # Take the winning move immediately

        # Next, check if the opponent can win in the next move and block it
        for action in valid_actions:
            state[action[0], action[1]] = self.opponent_number
            win, _ = check_win(state, action, self.opponent_number)
            state[action[0], action[1]] = 0
            if win:
                return action  # Block the opponent's winning move

        # Lookahead: Check if the opponent could win in two moves
        for action in valid_actions:
            state[action[0], action[1]] = self.player_number  # AI's move
            
            # Now simulate the opponent's response
            opponent_can_win = False
            for opponent_action in self.get_valid_actions(state):
                state[opponent_action[0], opponent_action[1]] = self.opponent_number  # Opponent's first move
                
                # Check if the opponent can set up a win on their next move
                for second_opponent_action in self.get_valid_actions(state):
                    state[second_opponent_action[0], second_opponent_action[1]] = self.opponent_number  # Opponent's second move
                    win, _ = check_win(state, second_opponent_action, self.opponent_number)
                    state[second_opponent_action[0], second_opponent_action[1]] = 0  # Reset

                    if win:
                        opponent_can_win = True
                        break  # No need to check further if opponent can win in two moves
                
                state[opponent_action[0], opponent_action[1]] = 0  # Reset
                
                if opponent_can_win:
                    break  # No need to check further if opponent can win in two moves

            state[action[0], action[1]] = 0  # Reset AI's move
            
            if opponent_can_win:
                return action  # Block the opponent's two-move winning setup

        # If no immediate or lookahead threats, evaluate the best move
        best_action = None
        best_score = float('-inf')
        for action in valid_actions:
            state[action[0], action[1]] = self.player_number
            current_score = self.evaluate_action(state, action, self.get_neighbours(dim, action), dim)
            state[action[0], action[1]] = 0
            
            if current_score > best_score:
                best_action = action
                best_score = current_score
        
        # Return the best action found
        return best_action

    def evaluate_action(self, state, action, neighbors, dim):
        """
        Evaluate the current action based on proximity to edges and corners, and 
        potential to connect or block a win condition, including forks.
        """
        score = 0
        
        # Increase score for proximity to edges
        if self.get_edge(action, dim) != -1:
            score += 10  # Edge positions are valuable
        
        # Increase score for proximity to corners
        if self.get_corner(action, dim) != -1:
            score += 15  # Corner positions are more valuable
        
        # Temporarily place the piece for evaluation
        state[action[0], action[1]] = self.player_number
        
        # Check for Fork, Bridge, and Ring potential
        win, structure = check_win(state, action, self.player_number)
        if win:
            if structure == "fork":
                score += 70  # Fork is very valuable, so give it a high score
            elif structure == "bridge":
                score += 50  # Bridge is valuable
            elif structure == "ring":
                score += 100  # Ring is the most valuable
        
        # Reset the state back after evaluation
        state[action[0], action[1]] = 0
        
        # Evaluate if this move can block an opponent's potential fork
        for neighbor in neighbors:
            if state[neighbor[0], neighbor[1]] == self.opponent_number:
                state[neighbor[0], neighbor[1]] = self.player_number  # Temporarily block
                potential_fork, _ = check_win(state, neighbor, self.opponent_number)
                state[neighbor[0], neighbor[1]] = self.opponent_number  # Reset back
                
                if potential_fork and _ == "fork":
                    score += 50  # Increase score for blocking opponent's potential fork

        return score

    # def evaluate_action(self, state, action, neighbors, dim):
    #     """
    #     Evaluate the current action based on proximity to edges and corners, and 
    #     potential to connect or block a win condition.
    #     """
    #     score = 0
        
    #     # Increase score for proximity to edges
    #     if self.get_edge(action, dim) != -1:
    #         score += 10  # Edge positions are valuable
        
    #     # Increase score for proximity to corners
    #     if self.get_corner(action, dim) != -1:
    #         score += 15  # Corner positions are more valuable
        
    #     # Temporarily place the piece for evaluation
    #     state[action[0], action[1]] = self.player_number
        
    #     # Check for Fork, Bridge, and Ring potential
    #     win, structure = check_win(state, action, self.player_number)
    #     if win:
    #         if structure == "fork":
    #             score += 50  # Fork is very valuable
    #         elif structure == "bridge":
    #             score += 40  # Bridge is also valuable
    #         elif structure == "ring":
    #             score += 100  # Ring is the most valuable
        
    #     # Reset the state back after evaluation
    #     state[action[0], action[1]] = 0
        
    #     return score


    def is_winning_state(self, state, player_number):
        """
        Check if the given player (player_num) is in a winning state.
        Win conditions include connecting opposite edges or forming a loop (ring).
        """
        dim = state.shape[0]
        
        # Get all positions where the player has placed their pieces
        player_positions = np.argwhere(state == player_number)
        
        # Check each position to see if it leads to a win
        for position in player_positions:
            if self.check_win_from_position(state, position, player_number, dim):
                return True
        
        return False

    def check_win_from_position(self, state, position, player_number, dim):
        """
        Check if a win condition is met starting from a given position.
        Uses DFS to explore connected paths for edge connection or ring formation.
        """
        stack = [(position, None)]  # (current_position, parent_position)
        visited = set()
        edges_reached = set()

        while stack:
            current_pos, parent_pos = stack.pop()
            
            if tuple(current_pos) in visited:
                continue
            
            visited.add(tuple(current_pos))
            
            # Check if the current position reaches an edge
            edge = self.get_edge(current_pos, dim)
            if edge != -1:
                edges_reached.add(edge)
                if self.check_opposite_edges_connected(edges_reached):
                    return True
            
            # Get neighbors of the current position
            neighbors = self.get_neighbours(dim, current_pos)
            
            for neighbor in neighbors:
                if state[neighbor[0], neighbor[1]] == player_number and (parent_pos is None or tuple(neighbor) != tuple(parent_pos)):
                    stack.append((neighbor, current_pos))
                    if self.check_ring(visited, neighbor, parent_pos):
                        return True
        
        return False

    def check_opposite_edges_connected(self, edges_reached):
        """
        Check if two opposite edges are connected:
        - 0 (top) and 2 (bottom)
        - 1 (left) and 3 (right)
        """
        return (0 in edges_reached and 2 in edges_reached) or (1 in edges_reached and 3 in edges_reached)

    def check_ring(self, visited, current_pos, parent_pos):
        """
        Check if a ring (loop) has been formed. A ring is detected when a position 
        is revisited, excluding the parent.
        """
        return tuple(current_pos) in visited and tuple(current_pos) != tuple(parent_pos)

    def get_valid_actions(self, state: np.array) -> List[Tuple[int, int]]:
        """
        Return a list of valid moves (unoccupied positions) on the board.
        """
        return list(zip(*np.where(state == 0)))  # Find all unoccupied (empty) positions

    def get_edge(self, position, dim):
        """
        Determine if a position is on an edge of the board:
        - 0: Top edge
        - 1: Left edge
        - 2: Bottom edge
        - 3: Right edge
        - -1: Not on an edge
        """
        x, y = position
        if x == 0:
            return 0  # Top edge
        elif x == dim - 1:
            return 2  # Bottom edge
        elif y == 0:
            return 1  # Left edge
        elif y == dim - 1:
            return 3  # Right edge
        return -1

    def get_corner(self, position, dim):
        """
        Determine if a position is a corner of the board:
        - Returns True if it's a corner, otherwise False.
        """
        corners = [(0, 0), (0, dim - 1), (dim - 1, 0), (dim - 1, dim - 1)]
        return position in corners

    def get_neighbours(self, dim, position):
        """
        Get the valid neighboring positions for a given position on the board.
        Neighbors are adjacent positions in the hex grid structure.
        """
        x, y = position
        neighbors = []
        
        possible_moves = [
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), 
            (x - 1, y + 1), (x + 1, y - 1)
        ]
        
        for nx, ny in possible_moves:
            if 0 <= nx < dim and 0 <= ny < dim:
                neighbors.append((nx, ny))
        
        return neighbors        
