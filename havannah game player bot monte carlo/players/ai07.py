#for now winning against ai2 both ways
import time
import random
import numpy as np
from helper import *

class AIPlayer:
    def __init__(self, player_number: int, timer, simulations=200, logging_mode="no"):
        self.player_number = player_number
        self.type = 'ai07'
        self.player_string = f'Player {player_number}: ai07'
        self.timer = timer
        self.opponent_number = 2 if player_number == 1 else 1
        self.simulations = simulations
        self.move_counter = 0
        self.board_dim = None
        self.visit_counts = np.zeros((10, 10))  # Adjust the dimensions as needed
        self.mcts_weights = np.zeros((10, 10))  # Assuming a 10x10 board
        self.logging_mode = logging_mode
        self.log_file = "log_timestamp.txt" if logging_mode == "file" else None

    def log(self, message):
        """Log messages based on the logging mode."""
        if self.logging_mode == "print":
            print(message)
        elif self.logging_mode == "file":
            with open(self.log_file, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    def get_move(self, state: np.array) -> Tuple[int, int]:
        """Decide the best move and explain the reasoning."""
        if self.board_dim is None:
            self.board_dim = state.shape[0]
            self.opening_moves = self.get_opening_moves(state)

        self.move_counter += 1

        # Early game strategy
        if self.move_counter <= len(self.opening_moves):
            for move in self.opening_moves[self.move_counter - 1:]:
                if self.is_valid_move(state, move):
                    self.log(f"Early game move to {move} as part of the opening strategy.")
                    return move

        valid_actions = get_valid_actions(state)

        # Check for immediate winning or blocking moves
        for action in valid_actions:
            if self.is_winning_move(state, action, self.player_number):
                self.log(f"Winning move detected at {action}.")
                return action
            if self.is_winning_move(state, action, self.opponent_number):
                self.log(f"Blocking opponent's winning move at {action}.")
                return action

        # Evaluate all valid moves and choose the best one
        best_score = float('-inf')
        best_actions = []

        for action in valid_actions:
            score = self.evaluate_move(state, action)
            self.log(f"Evaluating move {action}: score {score}.")
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        selected_action = random.choice(best_actions)
        self.log(f"Selected move {selected_action} with score {best_score}.")
        return selected_action

    def is_valid_move(self, state: np.array, move: Tuple[int, int]) -> bool:
        """Check if a move is valid."""
        return state[move[0], move[1]] == 0

    def is_winning_move(self, state, action, player_number):
        """Check if placing a piece results in a win."""
        state[action[0], action[1]] = player_number
        win, _ = check_win(state, action, player_number)
        state[action[0], action[1]] = 0
        return win

    def evaluate_move(self, state, action):
        """Evaluate the move by combining heuristic scores and simulations."""
        state[action[0], action[1]] = self.player_number

        # Calculate heuristic score based on several factors
        heuristic_score = self.assign_heuristic_score(state, action)
        self.log(f"Heuristic score for move {action}: {heuristic_score}.")

        # Monte Carlo simulations for mid-game and late-game decisions
        if self.move_counter > 2:
            mc_score = self.run_monte_carlo_simulations(state, action)
            self.log(f"Monte Carlo simulation score for move {action}: {mc_score}.")
            heuristic_score += mc_score

        # Adjust score based on the opponent's potential threats
        opponent_threat_score = self.detect_opponent_bridge_threat(state, action)
        self.log(f"Opponent threat score for move {action}: {opponent_threat_score}.")
        heuristic_score -= opponent_threat_score

        state[action[0], action[1]] = 0  # Reset state after evaluation
        return heuristic_score

    def assign_heuristic_score(self, state, action):
        """Calculate a heuristic score based on several strategic factors."""
        score = 0

        # Factor 1: Connectivity (friendly neighbors)
        friendly_neighbors = self.count_friendly_neighbors(state, action)
        score += friendly_neighbors * 5
        self.log(f"Connectivity score for move {action}: {friendly_neighbors * 5}.")

        # Factor 2: Blocking opponent's paths (opponent neighbors)
        opponent_neighbors = self.count_opponent_neighbors(state, action)
        score += opponent_neighbors * 3
        self.log(f"Blocking opponent paths score for move {action}: {opponent_neighbors * 3}.")

        # Factor 3: Pattern completion (bridge, fork, ring)
        pattern_score = self.evaluate_pattern_completion(state, action)
        score += pattern_score
        self.log(f"Pattern completion score for move {action}: {pattern_score}.")

        # Factor 4: Chain extension
        chain_length = self.get_chain_length(state, action)
        score += chain_length * 2
        self.log(f"Chain extension score for move {action}: {chain_length * 2}.")

        # Factor 5: Edge and corner control
        edge_score = self.evaluate_edge_control(action)
        score += edge_score
        self.log(f"Edge and corner control score for move {action}: {edge_score}.")

        # Factor 6: Assess future threats
        threat_score = self.assess_future_threats(state, action)
        score += threat_score
        self.log(f"Future threats score for move {action}: {threat_score}.")

        # Factor 7: Territorial control
        territorial_score = self.evaluate_territorial_control(action)
        score += territorial_score
        self.log(f"Territorial control score for move {action}: {territorial_score}.")

        return score


    def count_friendly_neighbors(self, state, action):
        """Count the number of friendly neighboring pieces."""
        count = 0
        neighbors = get_neighbours(self.board_dim, action)
        for neighbor in neighbors:
            if state[neighbor[0], neighbor[1]] == self.player_number:
                count += 1
        return count

    def count_opponent_neighbors(self, state, action):
        """Count the number of opponent neighboring pieces."""
        count = 0
        neighbors = get_neighbours(self.board_dim, action)
        for neighbor in neighbors:
            if state[neighbor[0], neighbor[1]] == self.opponent_number:
                count += 1
        return count

    def evaluate_pattern_completion(self, state, action):
        """Evaluate if the move completes any significant patterns (bridge, fork, ring)."""
        win, structure = check_win(state, action, self.player_number)
        if win:
            score = 500  # High value for completing a winning pattern
            print(f"Move {action} completes a {structure}, score {score}.")
            return score
        else:
            return 0

    def get_chain_length(self, state, action):
        """Determine the length of the chain connected through this move."""
        visited = set()
        chain_length = self.dfs(state, action, visited)
        return chain_length

    def dfs(self, state, position, visited):
        """Depth-First Search to count the connected pieces."""
        stack = [position]
        count = 0
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            count += 1
            neighbors = get_neighbours(self.board_dim, current)
            for neighbor in neighbors:
                if state[neighbor[0], neighbor[1]] == self.player_number and neighbor not in visited:
                    stack.append(neighbor)
        return count

    def evaluate_edge_control(self, action):
        """Evaluate control over edges and corners."""
        score = 0
        edge = get_edge(action, self.board_dim)
        corner = get_corner(action, self.board_dim)

        if self.move_counter <= 10:  # Early game
            if edge != -1:
                score += 5
            if corner != -1:
                score += 10
        else:  # Late game
            if edge != -1:
                score += 2
            if corner != -1:
                score += 3

        return score

    def assess_future_threats(self, state, action):
        """Assess potential future threats created by this move."""
        threat_level = self.count_friendly_neighbors(state, action)
        return threat_level * 5

    def evaluate_territorial_control(self, action):
        """Evaluate control over key areas, especially the center of the board."""
        center = self.board_dim // 2
        distance_to_center = abs(action[0] - center) + abs(action[1] - center)
        max_distance = center * 2
        return (max_distance - distance_to_center) * 2

    def run_monte_carlo_simulations(self, state, action):
        """Run Monte Carlo simulations to assess the move's potential."""
        win_count = 0
        loss_count = 0

        for _ in range(self.simulations):
            simulation_state = state.copy()
            simulation_state[action[0], action[1]] = self.player_number

            winner = self.simulate_random_game(simulation_state)
            if winner == self.player_number:
                win_count += 1
            elif winner == self.opponent_number:
                loss_count += 1

        return (win_count - loss_count) * 10  # Weight simulations to impact score

    def simulate_random_game(self, state):
        """Simulate a random game until a winner is found."""
        current_player = self.opponent_number if state.sum() % 2 == 0 else self.player_number
        while True:
            valid_actions = get_valid_actions(state)
            if not valid_actions:
                return 0  # Draw

            action = random.choice(valid_actions)
            state[action[0], action[1]] = current_player

            win, _ = check_win(state, action, current_player)
            if win:
                return current_player

            current_player = self.opponent_number if current_player == self.player_number else self.player_number

    def detect_opponent_bridge_threat(self, state, action):
        """Detect if the opponent is close to forming a bridge."""
        state[action[0], action[1]] = self.opponent_number  # Simulate opponent placing a piece
        if self.is_near_bridge(state, action):
            score = 500  # High threat score
            print(f"Move {action} blocks a near-bridge threat, threat score {score}.")
            state[action[0], action[1]] = 0
            return score

        state[action[0], action[1]] = 0  # Reset the simulated move
        return 0

    def is_near_bridge(self, state, action):
        """Check if the opponent is one move away from completing a bridge."""
        corners = get_all_corners(self.board_dim)
        state[action[0], action[1]] = self.opponent_number

        for i, corner1 in enumerate(corners):
            for j, corner2 in enumerate(corners):
                if i >= j:
                    continue
                if self.are_corners_connected(state, corner1, corner2):
                    state[action[0], action[1]] = 0
                    if not self.are_corners_connected(state, corner1, corner2):
                        state[action[0], action[1]] = self.opponent_number
                        return True

        state[action[0], action[1]] = 0
        return False

    def are_corners_connected(self, state, corner1, corner2):
        """Check if two corners are connected by a continuous line of opponent's pieces."""
        visited = set()
        stack = [corner1]

        while stack:
            current = stack.pop()
            if current == corner2:
                return True

            visited.add(current)
            neighbors = get_neighbours(self.board_dim, current)
            for neighbor in neighbors:
                if state[neighbor[0], neighbor[1]] == self.opponent_number and neighbor not in visited:
                    stack.append(neighbor)

        return False

    def get_opening_moves(self, state):
        """Return the center and nearby location of three unfilled corners."""
        dim = self.board_dim
        corners = {
            "top_left": (0, 0),
            "top_right": (0, dim-1),
            "bottom_left": (dim-1, 0),
            "bottom_right": (dim-1, dim-1)
        }
        
        # Check if specific corners are unfilled
        top_left_unfilled = self.is_valid_move(state, corners["top_left"])
        top_right_unfilled = self.is_valid_move(state, corners["top_right"])
        bottom_left_unfilled = self.is_valid_move(state, corners["bottom_left"])
        bottom_right_unfilled = self.is_valid_move(state, corners["bottom_right"])

        # If (0, 0), (0, dim-1), (dim-1, 0) are unfilled, return the specified sequence
        if top_left_unfilled and top_right_unfilled and bottom_left_unfilled:
            return [(0, dim-1), (1, dim-1)]
        
        # Fallback: return the center of the board as a default opening move
        center = (dim // 2, dim // 2)
        return [center]



