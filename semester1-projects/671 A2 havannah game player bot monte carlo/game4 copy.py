import os
import time
import json
import random
import multiprocessing as mp
from datetime import datetime
from multiprocessing import Value
from time import sleep
from threading import Thread 
from typing import Tuple, Callable, Dict
import numpy as np
import tkinter as tk

from helper import get_valid_actions, check_win, HEXAGON_COORDS, CLICK_EVENT, PLAYER_TIME
from players.ai01 import AIPlayer as AIPlayer01
from players.ai02 import AIPlayer as AIPlayer02
from players.ai03 import AIPlayer as AIPlayer03
from players.ai04 import AIPlayer as AIPlayer04
from players.ai05 import AIPlayer as AIPlayer05
from players.ai06 import AIPlayer as AIPlayer06
from players.ai07 import AIPlayer as AIPlayer07
from players.ai08 import AIPlayer as AIPlayer08
from players.ai09 import AIPlayer as AIPlayer09
from players.ai10 import AIPlayer as AIPlayer10
from players.ai11 import AIPlayer as AIPlayer11
from players.ai12 import AIPlayer as AIPlayer12
from players.ai13 import AIPlayer as AIPlayer13
from players.random import RandomPlayer

TimeLimitExceedAction = (1000, True)

def turn_worker(state: np.array, send_end, p_func: Callable[[np.array], Tuple[int, bool]], PLAYER_TIME):
    send_end.send(p_func(state, PLAYER_TIME))

def make_player(name, num, timer=PLAYER_TIME):
    if name == 'ai01':
        return AIPlayer01(num, timer)
    elif name == 'ai02':
        return AIPlayer02(num, timer)
    elif name == 'ai03':
        return AIPlayer03(num, timer)
    elif name == 'ai04':
        return AIPlayer04(num, timer)
    elif name == 'ai05':
        return AIPlayer05(num, timer)
    elif name == 'ai06':
        return AIPlayer06(num, timer)
    elif name == 'ai07':
        return AIPlayer07(num, timer)
    elif name == 'ai08':
        return AIPlayer08(num, timer)
    elif name == 'ai09':
        return AIPlayer09(num, timer)
    elif name == 'ai10':
        return AIPlayer10(num, timer)
    elif name == 'ai11':
        return AIPlayer11(num, timer)
    elif name == 'ai12':
        return AIPlayer12(num, timer)
    elif name == 'ai13':
        return AIPlayer13(num, timer)
    elif name == 'random':
        return RandomPlayer(num, timer)
    elif name == 'human':
        return HumanPlayer(num, timer)

class Game:
    def __init__(self, player1_name, player2_name, player1, player2, time: int, board_init: np.array, layers: int, mode: str):
        self.players = [player1, player2]
        self.colors = ['', 'yellow', 'red', 'black']
        self.faded_colors = ['', 'light yellow', 'orange', 'gray']
        self.layers = layers
        self.state = board_init
        self.gui_board = []
        PLAYER_TIME[0] = time
        PLAYER_TIME[1] = time
        self.use_gui = False
        self.structure_formed = None
        self.winning_path = []
        self.winner = None
        board = self.state

        self.current_turn = Value('i', 0)
        self.game_over = Value('b', False)
        self.pause_timer = Value('b', True)

        self.parent_conn, self.child_conn = mp.Pipe()
        self.proc = mp.Process(target=self.player_workers, args=(make_player, self.game_over, self.child_conn, player1_name, player2_name, PLAYER_TIME))
        self.proc.start()

        with open('logs.txt', 'w') as log_file:
            s = f'{layers}\n'
            for i in range(2 * layers - 1):
                for j in range(2 * layers - 1):
                    s += str(board[i][j]) + ' '
                s += '\n'
            log_file.write(s)
            log_file.write("Player 1 Type: " + player1.type + '\n')
            log_file.write("Player 2 Type: " + player2.type + '\n')
            print("Player 1 Type: " + player1.type + " Player 2 Type: " + player2.type)

        if mode == "gui":
            self.use_gui = True
            root = tk.Tk()
            root.title('Extended Havannah')

            self.current = tk.Label(root, text="Current:")
            self.current.pack()

            player1_string = f"{player1.player_string} (Yellow) | Time Remaining {PLAYER_TIME[0]:.2f} s"
            self.player1_string = tk.Label(root, text=player1_string, anchor="w", width=50)
            self.player1_string.pack()

            player2_string = f"{player2.player_string} (Red)    | Time Remaining {PLAYER_TIME[1]:.2f} s"
            self.player2_string = tk.Label(root, text=player2_string, anchor="w", width=50)
            self.player2_string.pack()

            self.scale = 1
            height = (25 * np.sqrt(3) * (2 * layers - 1)) * self.scale
            width = (75 * layers - 25) * self.scale
            self.c = tk.Canvas(root, height=height, width=width)
            self.c.pack()
            for j in range(2 * layers - 1):
                column = []
                col_size = layers
                if j < layers:
                    col_size += j
                else:
                    col_size += 2 * layers - 2 - j

                for i in range(col_size):
                    hex_coords = self.calculate_hexagon(i, j, 25, self.scale)
                    c = board[i][j]
                    self.display_coordinates(hex_coords, i, j)
                    hexagon_id = self.c.create_polygon(
                        hex_coords, fill=self.colors[c], outline="black")
                    HEXAGON_COORDS[hexagon_id] = (i, j)
                    column.append(hexagon_id)
                    self.c.tag_bind(hexagon_id, "<Button-1>", self.on_click)
                self.gui_board.append(column)

            timer = Thread(target=self.display_time, args=(self.game_over,))
            thread = Thread(target=self.threaded_function, args=(100000, self.game_over, self.pause_timer, self.current_turn))
            clock = mp.Process(target=self.clock, args=(self.game_over, self.pause_timer, self.current_turn, PLAYER_TIME))
            clock.start()
            timer.start()
            thread.start()
            root.mainloop()
        else:
            clock = mp.Process(target=self.clock, args=(self.game_over, self.pause_timer, self.current_turn, PLAYER_TIME))
            thread = Thread(target=self.threaded_function, args=(100000, self.game_over, self.pause_timer, self.current_turn))
            clock.start()
            thread.start()

    def player_workers(self, make_player, game_over, pipe_conn, player1, player2, timer):
        players = [make_player(player1, 1, timer), make_player(player2, 2, timer)]

        while not game_over.value:
            current_turn, state = pipe_conn.recv()
            move = players[current_turn].get_move(state)
            pipe_conn.send(move)

    def calculate_hexagon(self, i, j, size, scale=1):
        sqrt3 = np.sqrt(3)
        offset_x = j * size * 3 / 2
        offset_y = (abs(j - self.layers + 1) + 2 * i) * size * sqrt3 / 2
        return [
            ((size/2 + offset_x) * scale, offset_y * scale),
            ((size * 3/2 + offset_x) * scale, offset_y * scale),
            ((size * 2 + offset_x) * scale, (size * sqrt3/2 + offset_y) * scale),
            ((size * 3/2 + offset_x) * scale, (size * sqrt3 + offset_y) * scale),
            ((size / 2 + offset_x) * scale, (size * sqrt3 + offset_y) * scale),
            (offset_x * scale, (size * sqrt3/2 + offset_y) * scale)]

    def display_coordinates(self, hex_coords, i, j):
        x = sum([point[0] for point in hex_coords]) / 6
        y = sum([point[1] for point in hex_coords]) / 6
        self.c.create_text(x, y, text=f"({i},{j})", fill="black")

    def display_time(self, game_over):
        while not game_over.value:
            sleep(0.005)
            player1_string = f"{self.players[0].player_string} (Yellow) | Time Remaining {PLAYER_TIME[0]:.2f} s"
            player2_string = f"{self.players[1].player_string} (Red)    | Time Remaining {PLAYER_TIME[1]:.2f} s"
            self.player1_string.configure(text=player1_string, anchor="w", width=50)
            self.player2_string.configure(text=player2_string, anchor="w", width=50)

    def on_click(self, event):
        current_player = self.players[self.current_turn.value]
        if current_player.type == 'human':
            CLICK_EVENT[0] = event

    @staticmethod
    def clock(game_over, pause_timer, current_turn, PLAYER_TIME):
        start_time = [None, None]
        total_time = [PLAYER_TIME[0], PLAYER_TIME[1]]
        while not game_over.value:
            sleep(0.0001)
            if not pause_timer.value:
                if start_time[current_turn.value] is None:
                    start_time[current_turn.value] = time.time()
                    start_time[1 - current_turn.value] = None
                    total_time[current_turn.value] = PLAYER_TIME[current_turn.value]
                    continue

                if PLAYER_TIME[current_turn.value] > 0:
                    curr_time = time.time()
                    PLAYER_TIME[current_turn.value] = total_time[current_turn.value] + start_time[current_turn.value] - curr_time
                
                if PLAYER_TIME[current_turn.value] < 0:
                    PLAYER_TIME[current_turn.value] = 0.0

    def threaded_function(self, iterations, game_over, pause_timer, current_turn):
        sleep(1)  # Wait for tkinter to setup
        for _ in range(iterations):
            self.make_move(game_over, pause_timer, current_turn)
            sleep(0.01)

            if game_over.value:
                if self.proc.is_alive():
                    self.proc.terminate()

                with open('logs.txt', 'a') as log_file:
                    s = 'Game Over\n'
                    if self.use_gui:
                        for row, col in self.winning_path:
                            hex_coords = self.calculate_hexagon(row, col, 25, self.scale)
                            hex_coords.append(hex_coords[0])
                            self.c.create_line(hex_coords, fill="blue", width=5)

                        self.current.configure(text=f'GAME OVER\n Player {self.winner} won with a {self.structure_formed}', font=("Arial", 5 + self.state.shape[0], "bold"))

                    log_file.write(s)
                    log_file.write("Winner: Player " + str(self.winner) + '\n')
                    log_file.write("Structure Formed: " + str(self.structure_formed) + '\n')
                    log_file.write("Winning Path: " + str(self.winning_path) + '\n')
                    log_file.write("Player 1 Time Remaining: " + str(PLAYER_TIME[0]) + ' s\n')
                    log_file.write("Player 2 Time Remaining: " + str(PLAYER_TIME[1]) + ' s\n')
                    print(s)
                break

    def make_move(self, game_over, pause_timer, current_turn):
        current_player = self.players[current_turn.value]
        valid_actions = get_valid_actions(self.state, current_player.player_number)

        if len(valid_actions) == 0:
            game_over.value = True

        if not game_over.value:
            if current_player.type == 'ai':
                try:
                    pause_timer.value = False
                    self.parent_conn.send((current_turn.value, self.state))
                    if not self.parent_conn.poll(timeout=PLAYER_TIME[current_turn.value]):
                        game_over.value = True
                        self.winner = 2 - current_turn.value
                        pause_timer.value = True
                        raise Exception(f'Player {2 - current_turn.value} won!\nPlayer {current_turn.value + 1} exceeded time limit!')
                    action = self.parent_conn.recv()
                    pause_timer.value = True
                    action = int(action[0]), int(action[1])
                except Exception as e:
                    uh_oh = 'Uh oh.... something is wrong with Player {}'
                    print(uh_oh.format(current_player.player_number))
                    print(e)
                    action = TimeLimitExceedAction
            else:
                pause_timer.value = False
                action = current_player.get_move(self.state)
                pause_timer.value = True
                if (action == (-1, -1)) or (PLAYER_TIME[current_turn.value] < 0.001):
                    action = TimeLimitExceedAction
                    game_over.value = True
                    self.winner = 2 - current_turn.value
                    uh_oh = 'Uh oh.... something is wrong with Player {}'
                    print(uh_oh.format(current_player.player_number))
                    print(f'Player {2 - current_turn.value} won!\nPlayer {current_turn.value + 1} exceeded time limit!')

            if action == TimeLimitExceedAction:
                log_action = {'player': current_player.player_number, 'move': 'TLE'}
            elif action not in valid_actions:
                log_action = {'player': current_player.player_number, 'move': str(action) + ' is invalid'}
            else:
                move = action
                self.update_board(move, current_player.player_number, current_turn)
                log_action = {'player': current_player.player_number, 'move': move}

                self.winning_path = []
                win, way = check_win(self.state, move, current_player.player_number, self.winning_path)
                if win:
                    game_over.value = True
                    self.structure_formed = way
                    self.winner = current_player.player_number
                    print(f"GAME OVER, Player {self.winner} won with a {self.structure_formed}!")

            with open('logs.txt', 'a') as log_file:
                log_file.write(json.dumps(log_action, default=str) + '\n')
            current_turn.value = int(not current_turn.value)

            if self.use_gui:
                color = "Yellow" if current_turn.value == 0 else "Red"
                self.current.configure(text=f'Current Turn: {self.players[current_turn.value].player_string} ({color})')

    def update_board(self, cell: Tuple[int, int], player_num: int, current_turn):
        board = self.state
        row, col = cell
        if board[row, col] == 0:
            board[row, col] = player_num
            if self.use_gui:
                self.c.itemconfig(self.gui_board[col][row], fill=self.colors[current_turn.value + 1])
                hex_coords = self.calculate_hexagon(row, col, 25, self.scale)
                self.display_coordinates(hex_coords, row, col)
        else:
            err = 'Invalid move by player {}. Column {}'.format(player_num, cell)
            raise Exception(err)

def get_random_board(layers: int, blocks: int):
    assert layers > 1
    board = np.zeros([2 * layers - 1, 2 * layers - 1]).astype(np.uint8)
    for i in range(layers, 2 * layers - 1, 1):
        for j in range(0, i - layers + 1, 1):
            board[i][j] = 3
            board[i][2 * layers - 2 - j] = 3
    rand_x = np.random.randint(0, 2 * layers - 1, blocks)
    for x in rand_x:
        if x >= layers:
            y = np.random.randint(x - layers + 1, 3 * layers - 2 - x)
        else:
            y = np.random.randint(0, 2 * layers - 1)
        board[x][y] = 3
    return board

def main(n):
    player1 = 'ai'
    player2 = 'ai2'
    time = 240
    dim = 4
    blocks = 0
    mode = "server"

    # Counters for wins
    ai01_wins = 0
    ai02_wins = 0

    # Run n games
    for i in range(n):
        print(f"Running Game {i + 1}:")

        # Randomly decide the order of players for each game
        if i % 2 == 0:
            p1, p2 = 'ai', 'ai2'
        else:
            p1, p2 = 'ai2', 'ai'

        board = get_random_board(dim, blocks)
        game = Game(p1, p2, make_player(p1, 1), make_player(p2, 2), time, board, dim, mode)

        # Wait for the game process to finish
        game.proc.join()

        # Evaluate results after the game is completed
        if p1 == 'ai' and game.winner == 1:
            ai01_wins += 1
        elif p1 == 'ai2' and game.winner == 1:
            ai02_wins += 1
        elif p2 == 'ai' and game.winner == 2:
            ai01_wins += 1
        elif p2 == 'ai2' and game.winner == 2:
            ai02_wins += 1

    # Print summarized results after all games
    print(f"Final Results after {n} games: AI01 won {ai01_wins} games, AI02 won {ai02_wins} games.")

if __name__ == '__main__':
    n = 10  # Set the number of games you want to run
    main(n)


