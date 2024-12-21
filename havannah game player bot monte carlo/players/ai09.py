#for now winning against ai2 both ways
import time
import random
import numpy as np
from helper import *

class AIPlayer:
    def __init__(self, player_number: int, timer, simulations=100):
        self.player_number = player_number
        self.type = 'ai09'
        self.player_string = 'Player {}: ai09'.format(player_number)
        self.timer = timer
        self.opponent_number = 2 if player_number == 1 else 1
        self.simulations = simulations
        self.move_counter = 0
        self.opening_moves = []
        self.board_dim = None

        # MCTS-related attributes
        self.mcts_weights = np.zeros((10, 10))  # Example: storing weights based on a 10x10 board
        self.visit_counts = np.zeros((10, 10))  # Track how often a move was visited
        self.value_estimates = np.zeros((10, 10))  # Store value estimates for each move

    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Decide the best move for the current player by evaluating all valid moves,
        focusing on both blocking the opponent and creating winning opportunities.
        """
        # Set board dimension if not already set
        if self.board_dim is None:
            self.board_dim = state.shape[0]
            self.opening_moves = self.get_opening_moves()  # Initialize opening moves

        # Increment move counter
        self.move_counter += 1

        # Early game opening strategy
        if self.move_counter <= len(self.opening_moves):
            for move in self.opening_moves[self.move_counter - 1:]:
                if self.is_valid_move(state, move):
                    return move

        # Get the valid moves on the board (unoccupied positions)
        valid_actions = get_valid_actions(state)

        # First, check if there's a winning move for the AI
        for action in valid_actions:
            if self.is_winning_move(state, action, self.player_number):
                return action  # Take the winning move immediately

        # Check if the opponent can win in the next move and block it
        for action in valid_actions:
            if self.is_winning_move(state, action, self.opponent_number):
                return action  # Block the opponent's winning move

        # Evaluate all valid moves
        best_actions = []
        best_score = float('-inf')
        for action in valid_actions:
            current_score = self.evaluate_move(state, action)

            if current_score > best_score:
                best_score = current_score
                best_actions = [action]
            elif current_score == best_score:
                best_actions.append(action)

        # Return the best action found
        selected_action = random.choice(best_actions)
        self.update_mcts_weights(selected_action)

        return selected_action

    def is_valid_move(self, state: np.array, move: Tuple[int, int]) -> bool:
        """
        Check if a move is valid (i.e., the position is unoccupied).

        Parameters:
        state (np.array): The current state of the game board.
        move (Tuple[int, int]): The move to check.

        Returns:
        bool: True if the move is valid (unoccupied), False otherwise.
        """
        return state[move[0], move[1]] == 0

    def evaluate_move(self, state, action):
        """
        Evaluate the move by combining heuristic scores and simulations.
        """
        state[action[0], action[1]] = self.player_number

        # Calculate the heuristic score for the current move
        heuristic_score = self.assign_heuristic_score(state, action)

        # Only run Monte Carlo simulations if it's not one of the first three moves
        if self.move_counter > 2:
            heuristic_score += self.run_monte_carlo_simulations(state, action)

        # Check if the opponent could create a bridge or connect edges
        opponent_bridge_threat = self.detect_opponent_bridge_threat(state, action)
        heuristic_score -= opponent_bridge_threat  # Subtract the threat score to prioritize blocking

        state[action[0], action[1]] = 0  # Reset the AI's move

        return heuristic_score

    def detect_opponent_bridge_threat(self, state, action):
        """
        Detect if the opponent is close to forming a bridge and return a high threat score.
        """
        threat_score = 0
        state[action[0], action[1]] = self.opponent_number  # Simulate opponent placing a piece

        # Check for near-complete bridges or dual-edge threats
        if self.is_near_bridge(state, action) or self.is_dual_edge_threat(state, action):
            threat_score += 500  # Assign a high threat score for blocking

        state[action[0], action[1]] = 0  # Reset the simulated move
        return threat_score

    def is_near_bridge(self, state, action):
        """
        Check if the opponent is one move away from completing a bridge.
        """
        # Get the list of corners on the board
        corners = get_all_corners(self.board_dim)

        # Place the opponent's piece temporarily at the action position
        state[action[0], action[1]] = self.opponent_number

        # Iterate through each pair of corners
        for i, corner1 in enumerate(corners):
            for j, corner2 in enumerate(corners):
                if i >= j:
                    continue  # Skip redundant checks

                # Check if both corners are connected by opponent's pieces
                if self.are_corners_connected(state, corner1, corner2):
                    # If connected, remove the piece and check if it's a near-bridge
                    state[action[0], action[1]] = 0
                    if not self.are_corners_connected(state, corner1, corner2):
                        # Restore the piece and return True if it was a near-bridge
                        state[action[0], action[1]] = self.opponent_number
                        return True

        # Restore the board state
        state[action[0], action[1]] = 0
        return False

    def are_corners_connected(self, state, corner1, corner2):
        """
        Check if two corners are connected by a continuous line of opponent's pieces.
        """
        # Use BFS or DFS to check if there's a path between the two corners
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

    def is_dual_edge_threat(self, state, action):
        """
        Check if the opponent is close to connecting two edges, a common bridge setup.
        """
        edges_connected = set()
        neighbors = get_neighbours(self.board_dim, action)
        for neighbor in neighbors:
            if state[neighbor[0], neighbor[1]] == self.opponent_number:
                edge = get_edge(neighbor, self.board_dim)
                if edge != -1:
                    edges_connected.add(edge)
        return len(edges_connected) > 1  # True if more than one edge is connected

    def assign_heuristic_score(self, state, action):
        """
        Assign a heuristic score to a given action based on strategic importance.
        """
        score = 0
        dim = self.board_dim

        # Temporarily place the piece for evaluation
        state[action[0], action[1]] = self.player_number

        # 1. Connectivity Heuristic
        friendly_neighbors = self.count_friendly_neighbors(state, action)
        score += friendly_neighbors * 5  # Weight can be adjusted

        # 2. Blocking Opponent's Paths
        opponent_neighbors = self.count_opponent_neighbors(state, action)
        score += opponent_neighbors * 3  # Weight can be adjusted

        # 3. Pattern Completion
        pattern_score = self.evaluate_pattern_completion(state, action)
        score += pattern_score

        # 4. Chain Extension
        chain_length = self.get_chain_length(state, action)
        score += chain_length * 2  # Weight can be adjusted

        # 5. Edge and Corner Control Enhancement
        edge_score = self.evaluate_edge_control(action)
        score += edge_score

        # 6. Threat Level Assessment
        threat_score = self.assess_future_threats(state, action)
        score += threat_score

        # 7. Territorial Control
        territorial_score = self.evaluate_territorial_control(action)
        score += territorial_score

        # Reset the state back after evaluation
        state[action[0], action[1]] = 0

        return score

    def count_friendly_neighbors(self, state, action):
        """
        Count the number of friendly neighboring pieces.
        """
        count = 0
        neighbors = get_neighbours(self.board_dim, action)
        for neighbor in neighbors:
            if state[neighbor[0], neighbor[1]] == self.player_number:
                count += 1
        return count

    def count_opponent_neighbors(self, state, action):
        """
        Count the number of opponent neighboring pieces.
        """
        count = 0
        neighbors = get_neighbours(self.board_dim, action)
        for neighbor in neighbors:
            if state[neighbor[0], neighbor[1]] == self.opponent_number:
                count += 1
        return count

    def evaluate_pattern_completion(self, state, action):
        """
        Evaluate if the move completes any near-complete patterns.
        """
        score = 0
        win, structure = check_win(state, action, self.player_number)
        if win:
            if structure == "fork":
                score += 500  # High value for completing a Fork
            elif structure == "bridge":
                score += 500  # High value for completing a Bridge
            elif structure == "ring":
                score += 500  # High value for completing a Ring
        else:
            # Check for near-completion
            near_complete = self.check_near_completion(state, action)
            score += near_complete * 100  # Weight can be adjusted
        return score

    def check_near_completion(self, state, action):
        """
        Check for patterns that are one move away from completion.
        """
        # In a real implementation, you would check the board for near-complete structures
        near_complete = 0
        # Example logic for checking near-completion
        neighbors = get_neighbours(self.board_dim, action)
        for neighbor in neighbors:
            if state[neighbor[0], neighbor[1]] == self.player_number:
                # Add logic to check if this forms a near-complete structure (like a near-ring or near-fork)
                near_complete += 1  # Increment based on the structure found
        return near_complete

    def get_chain_length(self, state, action):
        """
        Get the length of the chain connected through this move.
        """
        visited = set()
        chain_length = self.dfs(state, action, visited)
        return chain_length

    def dfs(self, state, position, visited):
        """
        Depth-First Search to count the connected pieces.
        """
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
        """
        Adjust edge and corner control scores based on game progression.
        """
        score = 0
        dim = self.board_dim
        edge = get_edge(action, dim)
        corner = get_corner(action, dim)

        # Early game: Edges and corners are more valuable
        if self.move_counter <= 10:
            if edge != -1:
                score += 5  # Increased weight for edges
            if corner != -1:
                score += 10  # Increased weight for corners
        else:
            # Late game: Edges and corners are less valuable
            if edge != -1:
                score += 2
            if corner != -1:
                score += 3
        return score

    def assess_future_threats(self, state, action):
        """
        Assess the potential threats created by this move.
        """
        threat_level = self.count_friendly_neighbors(state, action)
        return threat_level * 5  # Weight can be adjusted

    def evaluate_territorial_control(self, action):
        """
        Evaluate control over key areas of the board.
        """
        dim = self.board_dim
        center = dim // 2
        distance_to_center = abs(action[0] - center) + abs(action[1] - center)
        max_distance = center * 2
        # Closer to center means higher score
        control_score = (max_distance - distance_to_center) * 2  # Weight can be adjusted
        return control_score

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

            current_player = self.opponent_number if current_player == self.player_number else self.player_number

    def get_opening_moves(self):
        """
        Return a predefined list of strong opening moves.
        """
        center = self.board_dim // 2
        # Example opening moves (should be customized based on strategy)
        return [
            (center, center),
            (center, center - 1)
        ]

    def update_mcts_weights(self, action: Tuple[int, int]):
        """Update MCTS weights and visit counts."""
        x, y = action
        self.visit_counts[x, y] += 1
        learning_rate = 1 / (self.visit_counts[x, y] + 1)
        self.mcts_weights[x, y] += learning_rate * (1 - self.mcts_weights[x, y])

    def self_play(self, num_games=10):
        """Let the AI play against itself to improve MCTS weights."""
        for _ in range(num_games):
            state = np.zeros((self.board_dim, self.board_dim))
            winner = None
            move_sequence = []

            while not winner:
                move = self.get_move(state)
                state[move[0], move[1]] = self.player_number
                move_sequence.append(move)
                winner, _ = check_win(state, move, self.player_number)
                if not winner:
                    self.player_number, self.opponent_number = self.opponent_number, self.player_number

            # Update weights based on the outcome of the game
            self.update_mcts_post_game(winner, move_sequence)

    def update_mcts_post_game(self, winner, move_sequence):
        """Update MCTS weights based on the outcome of the game."""
        reward = 1 if winner == self.player_number else -1
        for move in move_sequence:
            x, y = move
            self.value_estimates[x, y] += reward
            reward *= -1  # Flip the reward for alternating moves

    def is_winning_move(self, state, action, player_number):
        """Check if the move results in a win."""
        state[action[0], action[1]] = player_number
        win, _ = check_win(state, action, player_number)
        state[action[0], action[1]] = 0
        return win
