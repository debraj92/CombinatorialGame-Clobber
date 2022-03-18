import random

from SubgameCache import clobberSubgames
from game_basics import EMPTY, BLACK, WHITE, isEmptyBlackWhite, opponent
import time

import keras.models
import numpy as np


class Clobber_1d(object):
    board_features = None
    MAX_LENGTH_FEATURES = 40
    winning_black_positions = dict()
    winning_white_positions = dict()
    n_positions = set()
    p_positions = dict()

    # Board is stored in 1-d array of EMPTY, BLACK, WHITE

    player_positions = set()
    opponent_positions = set()
    first_player = WHITE
    opponent_player = BLACK

    def custom_board(self, start_position):  # str of B, W, E or .
        color_map = {"B": BLACK, "W": WHITE, "E": EMPTY, ".": EMPTY}
        board = []
        for c in start_position:
            board.append(color_map[c])

        return board

    @classmethod
    def to_string(cls, board):
        char_map = {BLACK: "B", WHITE: "W", EMPTY: "."}
        s = ""
        for p in board:
            s += char_map[p]
        return s

    def isCurrentPlayerWhite(self):
        return self.toPlay == WHITE

    def isCurrentPlayerBlack(self):
        return self.toPlay == BLACK

    def updatePositions(self):
        for i, p in enumerate(self.board):
            if p == self.first_player:
                self.player_positions.add(i)
            elif p == self.opponent_player:
                self.opponent_positions.add(i)

    def initZobristHashTable(self, HashSeed=None):
        for i in range(len(self.board)):
            random_values = []
            if HashSeed is not None:
                random.seed(HashSeed)
            else:
                random.seed(time.time())
            for j in range(3):
                random_values.append(random.getrandbits(64))

            self.hash_table.append(random_values)

    def __init__(self, start_position, first_player=WHITE, HashSeed=None):
        self.first_player = first_player
        self.setOpponentPlayer()
        self.board_features = None
        self.board = self.custom_board(start_position)
        self.resetGame(first_player)
        self.updatePositions()
        self.hash_table = []
        self.board_hash_value = 0
        self.initZobristHashTable(HashSeed)
        self.getBoardHash()

        self.PROVEN_WIN = 10000
        self.PROVEN_LOSS = -10000
        self.UNKNOWN = -5

        gameCache = clobberSubgames()
        self.winning_black_positions = gameCache.winning_black_positions
        self.winning_white_positions = gameCache.winning_white_positions
        self.p_positions = gameCache.p_positions
        self.n_positions = gameCache.n_positions

    def __hash__(self):
        return self.getBoardHash()

    def __eq__(self, other):
        return self.getBoardHash() == other.getBoardHash()

    def setOpponentPlayer(self):
        if self.first_player == WHITE:
            self.opponent_player = BLACK
        else:
            self.opponent_player = WHITE

    def isFirstPlayer(self):
        return self.toPlay == self.first_player

    def isOpponent(self):
        return self.toPlay == self.opponent_player

    def resetGame(self, first_player):
        self.toPlay = first_player
        self.opponent_positions = set()
        self.player_positions = set()

    def opp_color(self):
        if self.toPlay == WHITE:
            return BLACK
        return WHITE

    def switchToPlay(self):
        self.toPlay = self.opp_color()

    def sizeOfRemainingGame(self, depth):
        return len(self.board) - depth

    def updateBoardHashValue(self, move):
        savedHashValue = self.board_hash_value
        src, to = move
        self.board_hash_value ^= self.hash_table[src][self.toPlay]
        self.board_hash_value ^= self.hash_table[to][self.opp_color()]

        self.board_hash_value ^= self.hash_table[src][EMPTY]
        self.board_hash_value ^= self.hash_table[to][self.toPlay]
        return savedHashValue

    def updateFeatureAtIndex(self, idx, feature):
        if feature == EMPTY:
            self.board_features[idx][0] = 0
            self.board_features[idx][1] = 0
        elif feature == BLACK:
            self.board_features[idx][0] = 0
            self.board_features[idx][1] = 1
        else:
            self.board_features[idx][0] = 1
            self.board_features[idx][1] = 0

    def play(self, move):
        src, to = move
        if self.toPlay == self.first_player:
            self.player_positions.add(to)
            self.player_positions.remove(src)
            self.opponent_positions.remove(to)
        else:
            self.opponent_positions.add(to)
            self.opponent_positions.remove(src)
            self.player_positions.remove(to)
        self.board[src] = EMPTY
        self.board[to] = self.toPlay

        savedHashValue = self.updateBoardHashValue(move)
        self.switchToPlay()
        return savedHashValue

    def undoMove(self, move, savedHashValue):
        self.switchToPlay()
        src, to = move
        self.board[to] = self.opp_color()
        self.board[src] = self.toPlay

        self.board_hash_value = savedHashValue
        if self.toPlay == self.first_player:
            self.player_positions.add(src)
            self.player_positions.remove(to)
            self.opponent_positions.add(to)
        else:
            self.opponent_positions.add(src)
            self.opponent_positions.remove(to)
            self.player_positions.add(to)

    def applyMoveForFeatureEvaluation(self, move):
        src, to = move
        self.updateFeatureAtIndex(src, EMPTY)
        self.updateFeatureAtIndex(to, self.toPlay)

    def undoMoveFromFeatureEvaluation(self, move):
        src, to = move
        self.updateFeatureAtIndex(to, self.opp_color())
        self.updateFeatureAtIndex(src, self.toPlay)

    def winner(self, isEndOfGame):
        if isEndOfGame:
            return self.opp_color()
        else:
            return EMPTY

    def staticallyEvaluateForToPlay(self, isEndOfGame):
        winColor = self.winner(isEndOfGame)
        if winColor == self.toPlay:
            return self.PROVEN_WIN
        elif winColor == EMPTY:
            return self.UNKNOWN
        else:
            return self.PROVEN_LOSS

    def printPositions(self):
        print("Players in ", self.player_positions)
        print("Opponents in ", self.opponent_positions)

    def print_(self):
        print(Clobber_1d.to_string(self.board))

    def getBoardHash(self):
        if self.board_hash_value > 0:
            return self.board_hash_value
        zobristHashValue = 0
        for i, p in enumerate(self.board):
            zobristHashValue ^= self.hash_table[i][p]

        self.board_hash_value = zobristHashValue
        return self.board_hash_value

    def isCNNMoveOrderingActive(self, depth, score):
        return (score > -0.9) and self.sizeOfRemainingGame(depth) <= 17

    def computePrunedMovesFromSubgames(self, previous_score, depth):

        games = dict()
        BW = BLACK + WHITE
        moves_subgame = set()
        current_game = ""
        inverse_game = ""
        reversed_inverse_game = ""
        isfirstPlayer = self.toPlay == self.first_player
        opp = self.opp_color()
        last = len(self.board) - 1
        positions = self.player_positions
        isCnnActive = self.isCNNMoveOrderingActive(depth, previous_score)
        if isCnnActive:
            self.board_features = np.empty(shape=[0, 2], dtype=np.float32)

        flips = 0
        runningColor = None
        flip_left_side = 0
        flip_right_side = 0
        if not isfirstPlayer:
            positions = self.opponent_positions

        if self.toPlay == BLACK:
            winning_boards = self.winning_black_positions
            losing_boards = self.winning_white_positions
        else:
            winning_boards = self.winning_white_positions
            losing_boards = self.winning_black_positions

        for i, p in enumerate(self.board):
            if self.board[i] != EMPTY:
                if isCnnActive:
                    if self.board[i] == 1:
                        self.board_features = np.append(self.board_features, np.array([[0, 1]], dtype=np.float32), axis=0)
                    else:
                        self.board_features = np.append(self.board_features, np.array([[1, 0]], dtype=np.float32), axis=0)

                if flips <= 2:
                    if runningColor is None or runningColor != self.board[i]:
                        flips += 1
                        runningColor = self.board[i]
                    if flips == 1:
                        flip_left_side += 1
                    elif flips == 2:
                        flip_right_side += 1
                current_game += str(self.board[i])
                inverse_game += str(BW - self.board[i])
                reversed_inverse_game = str(BW - self.board[i]) + reversed_inverse_game
                if i in positions:
                    if i > 0 and self.board[i - 1] == opp:
                        moves_subgame.add((i, i - 1))
                    if i < last and self.board[i + 1] == opp:
                        moves_subgame.add((i, i + 1))
            else:

                if isCnnActive and i > 0 and self.board[i-1] != EMPTY:
                    self.board_features = np.append(self.board_features, np.array([[0, 0]], dtype=np.float32), axis=0)

                isZero = (flips == 2 and flip_left_side >= 2 and flip_right_side >= 2)
                if len(current_game) > 0:
                    if (current_game == "12" or current_game == "21") and current_game in games:
                        games.pop(current_game)
                    elif inverse_game in games:
                        games.pop(inverse_game)
                    elif reversed_inverse_game in games:
                        games.pop(reversed_inverse_game)
                    elif not isZero:
                        totalMoves = len(moves_subgame)
                        if totalMoves > 0 and current_game not in self.p_positions:
                            depth_L = winning_boards.get(current_game)
                            iswinning = depth_L is not None
                            islosing = current_game in losing_boards
                            isN = current_game in self.n_positions

                            if iswinning:
                                sortKey = -depth_L
                            else:
                                sortKey = totalMoves

                            games[current_game] = (moves_subgame, sortKey, iswinning, islosing, isN)

                    current_game = ""
                    inverse_game = ""
                    reversed_inverse_game = ""
                    moves_subgame = set()
                    flips = 0
                    runningColor = None
                    flip_left_side = 0
                    flip_right_side = 0

        isZero = (flips == 2 and flip_left_side >= 2 and flip_right_side >= 2)
        if len(current_game) > 0:
            if (current_game == "12" or current_game == "21") and current_game in games:
                games.pop(current_game)
            elif inverse_game in games:
                games.pop(inverse_game)
            elif reversed_inverse_game in games:
                games.pop(reversed_inverse_game)
            elif not isZero:
                totalMoves = len(moves_subgame)
                if totalMoves > 0 and current_game not in self.p_positions:
                    depth_L = winning_boards.get(current_game)
                    iswinning = depth_L is not None
                    islosing = current_game in losing_boards
                    isN = current_game in self.n_positions

                    if iswinning:
                        sortKey = -depth_L
                    else:
                        sortKey = totalMoves

                    games[current_game] = (moves_subgame, sortKey, iswinning, islosing, isN)

        if isCnnActive:

            empty_positions_to_add = self.MAX_LENGTH_FEATURES - len(self.board_features)
            dots = np.full((empty_positions_to_add, 2), [[0, 0]], dtype=np.float32)
            self.board_features = np.concatenate((self.board_features, dots), axis=0)

        return games.values()
