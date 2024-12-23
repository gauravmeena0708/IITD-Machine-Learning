#for now winning against ai2 but sometimes losing as well
import time
import random
import numpy as np
from helper import *

class AIPlayer:

    def __init__(self, player_number: int, timer, simulations=100):
        """
        Initialize the AIPlayer Agent

        Parameters:
        player_number (int): Current player number (1 or 2)
        timer: Timer object to fetch the remaining time for any player
        simulations (int): Number of Monte Carlo simulations per move
        """
        self.player_number = player_number
        self.type = 'ai06'
        self.player_string = 'Player {}: ai06'.format(player_number)
        self.timer = timer
        self.opponent_number = 2 if player_number == 1 else 1
        self.simulations = simulations
        self.move_counter = 0  # Initialize move counter
        self.opening_moves = self.get_opening_moves()  # Predefined opening moves

    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Decide the best move for the current player by evaluating all valid moves,
        focusing on both blocking the opponent and creating winning opportunities.
        """
        # Increment move counter
        self.move_counter += 1

        # Early game opening strategy
        if self.move_counter <= len(self.opening_moves):
            return self.opening_moves[self.move_counter - 1]

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
            max_opponent_threat_score = self.evaluate_opponent_threat(state, opponent_moves)

            # Calculate the heuristic score for the current move
            current_score = self.assign_heuristic_score(state, action)

            # Only run Monte Carlo simulations if it's not one of the first three moves
            if self.move_counter > 3:
                current_score += self.run_monte_carlo_simulations(state, action)

            # Subtract the opponent's best response threat from the AI's evaluation
            current_score -= max_opponent_threat_score
            state[action[0], action[1]] = 0  # Reset the AI's move

            if current_score > best_score:
                best_score = current_score
                best_actions = [action]
            elif current_score == best_score:
                best_actions.append(action)

        # Use alpha-beta pruning or dynamic time allocation if game is in mid-to-late phase
        if self.move_counter > 10:
            best_actions = self.alpha_beta_search(state, best_actions)

        # Return the best action found
        return random.choice(best_actions)

    def evaluate_opponent_threat(self, state, opponent_moves):
        """
        Evaluate the potential threat from opponent's moves and return the maximum threat score.
        """
        max_opponent_threat_score = 0
        for opponent_move in opponent_moves:
            state[opponent_move[0], opponent_move[1]] = self.opponent_number
            win, _ = check_win(state, opponent_move, self.opponent_number)
            if win:
                max_opponent_threat_score = max(max_opponent_threat_score, 100)  # High score for opponent's winning move
            else:
                # Now simulate the AI's move after the opponent's move
                second_step_valid_actions = get_valid_actions(state)
                for second_step_action in second_step_valid_actions:
                    state[second_step_action[0], second_step_action[1]] = self.player_number
                    second_step_win, _ = check_win(state, second_step_action, self.player_number)
                    if second_step_win:
                        max_opponent_threat_score = max(max_opponent_threat_score, -50)  # Reward for AI setting up a win
                    state[second_step_action[0], second_step_action[1]] = 0
            state[opponent_move[0], opponent_move[1]] = 0  # Reset the opponent's move

        return max_opponent_threat_score

    def assign_heuristic_score(self, state, action):
        """
        Assign a heuristic score to a given action based on strategic importance.
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

    def run_monte_carlo_simulations(self, state, action):
        """
        Run Monte Carlo simulations from the given state and action to update the heuristic score.
        """
        win_count = 0
        loss_count = 0

        for _ in range(self.simulations):
            simulation_state = state.copy()
            simulation_state[action[0], action[1]] = self.player_number

            # Simulate random playouts
            winner = self.simulate_random_game(simulation_state)

            if winner == self.player_number:
                win_count += 1
            elif winner == self.opponent_number:
                loss_count += 1

        # Return the net score from simulations
        return (win_count - loss_count) * 10  # Weight the result by 10 to adjust its impact on the heuristic

    def simulate_random_game(self, state):
        """
        Simulate a random game from the given state until a win condition is met.
        """
        current_player = self.opponent_number if state.sum() % 2 == 0 else self.player_number
        while True:
            valid_actions = get_valid_actions(state)
            if not valid_actions:
                return 0  # Draw if no valid actions left

            action = random.choice(valid_actions)
            state[action[0], action[1]] = current_player

            win, _ = check_win(state, action, current_player)
            if win:
                return current_player

            # Switch players
            current_player = self.opponent_number if current_player == self.player_number else self.player_number

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

    def alpha_beta_search(self, state, best_actions):
        """
        Perform alpha-beta pruning to refine the selection of best actions.
        """
        alpha = float('-inf')
        beta = float('inf')
        best_value = float('-inf')
        best_move = None

        for action in best_actions:
            value = self.minimax(state, action, depth=3, alpha=alpha, beta=beta, maximizing_player=True)
            if value > best_value:
                best_value = value
                best_move = action

        return [best_move] if best_move else best_actions

    def minimax(self, state, action, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning.
        """
        if depth == 0 or self.is_terminal(state):
            return self.assign_heuristic_score(state, action)

        if maximizing_player:
            max_eval = float('-inf')
            valid_actions = get_valid_actions(state)
            for new_action in valid_actions:
                state[new_action[0], new_action[1]] = self.player_number
                eval = self.minimax(state, new_action, depth - 1, alpha, beta, False)
                state[new_action[0], new_action[1]] = 0  # Undo move
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            valid_actions = get_valid_actions(state)
            for new_action in valid_actions:
                state[new_action[0], new_action[1]] = self.opponent_number
                eval = self.minimax(state, new_action, depth - 1, alpha, beta, True)
                state[new_action[0], new_action[1]] = 0  # Undo move
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def is_terminal(self, state):
        """
        Check if the game is in a terminal state (win/loss/draw).
        """
        return len(get_valid_actions(state)) == 0 or self.check_for_win(state)

    def check_for_win(self, state):
        """
        Check if there is a win condition on the board.
        """
        for action in get_valid_actions(state):
            if check_win(state, action, self.player_number)[0] or check_win(state, action, self.opponent_number)[0]:
                return True
        return False

    def get_opening_moves(self):
        """
        Return a predefined list of strong opening moves.
        """
        return [(0, 0), (0, 1), (1, 0)]  # Example moves, customize as needed
