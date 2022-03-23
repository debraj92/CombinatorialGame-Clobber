import random
import sys

sys.path.append("../")
from game_basics import EMPTY, BLACK, WHITE


class ClobberEnvironment:
    def __init__(self, maximum_board_size):
        self.maximum_board_size = maximum_board_size
        self.action_map = self.generate_action_map()
        self.legal_moves = []
        self.reset()

    def step(self, action):
        # Execute one time step within the environment
        if action not in self.legal_moves:
            raise ValueError(
                f"Illegal action.\nBoard: {self.board}\nAction: {action}\nPlayer: {self.current_player}"
            )
        self.play_move(action)
        self.compute_legal_moves()
        end_of_game = self.is_end_of_game()
        reward = self.compute_rewards(end_of_game)
        return self.board, self.current_player, reward, end_of_game

    def reset(self):
        # Reset the state of the environment to an initial state
        while self.is_end_of_game():
            self.board = self.generate_board()
            self.first_player = BLACK # random.choice([BLACK, WHITE])
            self.current_player = self.first_player
            self.compute_legal_moves()
        return self.board, self.current_player

    def reset_to_board(self, board, first_player):
        self.board = board
        self.first_player = first_player
        self.current_player = first_player
        self.compute_legal_moves()
        return self.board, self.current_player

    def compute_rewards(self, end_of_game):
        if end_of_game:
            if self.current_player == self.first_player:
                return -1
            else:
                return +1
        else:
            return -1 / (100 * self.maximum_board_size)

    def generate_action_map(self):
        # Generates the set of all possible actions for the board
        all_moves = []
        for i in range(self.maximum_board_size):
            left = i
            right = i + 1
            if right < self.maximum_board_size:
                all_moves.append((left, right))
                all_moves.append((right, left))
        return {key: value for value, key in enumerate(sorted(all_moves))}

    def generate_board(self):
        # Returns a randomly generated board of maximum board size
        # 66% of the time it returns a full board - no empty spots
        if random.choice([True, True, False]):
            return random.choices([BLACK, WHITE], k=self.maximum_board_size)
        else:
            return random.choices([BLACK, WHITE, EMPTY], k=self.maximum_board_size)

    def compute_legal_moves(self):
        moves = []
        last = len(self.board) - 1
        opponent_player = self.opponent_color()
        for i, p in enumerate(self.board):
            if p == self.current_player:
                if i > 0 and self.board[i - 1] == opponent_player:
                    moves.append((i, i - 1))
                if i < last and self.board[i + 1] == opponent_player:
                    moves.append((i, i + 1))
        self.legal_moves = moves

    def get_legal_moves(self):
        return self.legal_moves

    def get_action_map(self):
        return self.action_map

    def play_move(self, move):
        # Make move
        src, to = move
        self.board[src] = EMPTY
        self.board[to] = self.current_player
        # Switch current player
        self.switch_current_player()

    def opponent_color(self):
        if self.current_player == WHITE:
            return BLACK
        return WHITE

    def switch_current_player(self):
        self.current_player = self.opponent_color()

    def is_end_of_game(self):
        return len(self.legal_moves) == 0

    def board_to_string(self):
        char_map = {BLACK: "B", WHITE: "W", EMPTY: "."}
        s = ""
        for p in self.board:
            s += char_map[p]
        return s
