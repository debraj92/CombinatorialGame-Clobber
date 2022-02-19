import random
from game_basics import EMPTY, BLACK, WHITE, isEmptyBlackWhite, opponent
import time


class Clobber_1d(object):
    # Board is stored in 1-d array of EMPTY, BLACK, WHITE

    player_positions = set()
    opponent_positions = set()
    first_player = WHITE
    opponent_player = BLACK

    @classmethod
    def standard_board(cls, size):
        pairs = (size + 1) // 2
        board = [BLACK, WHITE] * pairs
        return board[:size]

    @classmethod
    def custom_board(cls, start_position):  # str of B, W, E or .
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
        # we take either a board size for standard "BWBW...",
        # or a custom start string such as "BWEEWWB"
        self.first_player = first_player
        self.setOpponentPlayer()
        if type(start_position) == int:
            self.init_board = Clobber_1d.standard_board(start_position)
        else:
            self.init_board = Clobber_1d.custom_board(start_position)
        self.resetGame(first_player)
        self.updatePositions()
        self.hash_table = []
        self.board_hash_value = 0
        self.initZobristHashTable(HashSeed)
        self.getBoardHash()

        self.PROVEN_WIN = 10000
        self.PROVEN_LOSS = -10000
        self.UNKNOWN = -5

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
        self.board = self.init_board
        self.toPlay = first_player
        self.opponent_positions = set()
        self.player_positions = set()

    def opp_color(self):
        if self.toPlay == WHITE:
            return BLACK
        return WHITE

    def switchToPlay(self):
        self.toPlay = self.opp_color()

    def updateBoardHashValue(self, move):
        savedHashValue = self.board_hash_value
        src, to = move
        self.board_hash_value ^= self.hash_table[src][self.toPlay]
        self.board_hash_value ^= self.hash_table[to][self.opp_color()]

        self.board_hash_value ^= self.hash_table[src][EMPTY]
        self.board_hash_value ^= self.hash_table[to][self.toPlay]
        return savedHashValue

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

    def computeLegalMoves(self):
        isfirstPlayer = self.toPlay == self.first_player
        moves = set()
        opp = self.opp_color()
        last = len(self.board) - 1
        positions = self.player_positions
        if not isfirstPlayer:
            positions = self.opponent_positions

        for position in positions:
            if position > 0 and self.board[position - 1] == opp:
                moves.add((position, position - 1))
            if position < last and self.board[position + 1] == opp:
                moves.add((position, position + 1))

        return moves

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

    def pruneMovesUsingSecondPlayerWin(self):

        games = dict()
        BW = BLACK + WHITE
        moves_subgame = set()
        foundZeroSum = False
        current_game = ""
        inverse_game = ""
        unpruned_moves = set()

        isfirstPlayer = self.toPlay == self.first_player
        opp = self.opp_color()
        last = len(self.board) - 1
        positions = self.player_positions
        if not isfirstPlayer:
            positions = self.opponent_positions

        for i, p in enumerate(self.board):
            if self.board[i] != EMPTY:
                current_game += str(self.board[i])
                inverse_game += str(BW - self.board[i])
                if i in positions:
                    if i > 0 and self.board[i - 1] == opp:
                        moves_subgame.add((i, i - 1))
                        unpruned_moves.add((i, i - 1))
                    if i < last and self.board[i + 1] == opp:
                        moves_subgame.add((i, i + 1))
                        unpruned_moves.add((i, i + 1))
            else:
                if len(current_game) > 0:
                    if current_game in games:
                        games.pop(current_game)
                        foundZeroSum = True
                    elif inverse_game in games:
                        games.pop(inverse_game)
                        foundZeroSum = True
                    else:
                        if len(moves_subgame) > 0:
                            games[current_game] = moves_subgame

                    current_game = ""
                    inverse_game = ""
                    moves_subgame = set()

        if len(current_game) > 0:
            if current_game in games:
                games.pop(current_game)
                foundZeroSum = True
            elif inverse_game in games:
                games.pop(inverse_game)
                foundZeroSum = True
            else:
                if len(moves_subgame) > 0:
                    games[current_game] = moves_subgame

        if foundZeroSum:
            allowedMoves = set()
            for _, moves in games.items():
                allowedMoves.update(moves)
            return allowedMoves
        else:
            return unpruned_moves

