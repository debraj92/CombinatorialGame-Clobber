import json
import time

from game_basics import EMPTY, BLACK, WHITE
from boolean_negamax_tt import PlayClobber
from clobber_1d import Clobber_1d
from search_basics import INFINITY, PROVEN_WIN, PROVEN_LOSS, UNKNOWN


class EndGameDB:
    players = ["B", "W"]

    games = set()

    N = set()
    P = dict()
    WINING_B = dict()
    WINING_W = dict()

    def generateGameCombinations(self, boardString, length, maxlength):

        if length == maxlength:
            self.games.add(boardString)
            return

        for i in self.players:
            self.generateGameCombinations(boardString + i, length + 1, maxlength)

    def encodeBoard(self, game):
        game = game.replace("B", "1")
        game = game.replace("W", "2")
        return game

    def createLRclasses(self):

        for game in self.games:

            first_player = BLACK
            clobber = Clobber_1d(game, first_player)
            play = PlayClobber()
            start_time = time.time()
            b_first_result, _, _ = play.negamaxClobberGamePlay(clobber, start_time)

            if b_first_result == PROVEN_WIN:

                first_player = WHITE
                clobber = Clobber_1d(game, first_player)
                play = PlayClobber()
                start_time = time.time()
                w_first_result, _, _ = play.negamaxClobberGamePlay(clobber, start_time)

                if w_first_result == PROVEN_LOSS:
                    self.WINING_B[self.encodeBoard(game)] = play.getMaxDepth()
            else:

                first_player = WHITE
                clobber = Clobber_1d(game, first_player)
                play = PlayClobber()
                start_time = time.time()
                w_first_result, _, _ = play.negamaxClobberGamePlay(clobber, start_time)

                if w_first_result == PROVEN_WIN:
                    self.WINING_W[self.encodeBoard(game)] = play.getMaxDepth()

    def createNPpositions(self):
        for game in self.games:

            first_player = BLACK
            clobber = Clobber_1d(game, first_player)
            play = PlayClobber()
            start_time = time.time()
            b_first_result, _, _ = play.negamaxClobberGamePlay(clobber, start_time)
            max_depth = play.getMaxDepth()
            first_player = WHITE
            clobber = Clobber_1d(game, first_player)
            play = PlayClobber()
            start_time = time.time()
            w_first_result, _, _ = play.negamaxClobberGamePlay(clobber, start_time)

            if b_first_result == PROVEN_WIN and w_first_result == PROVEN_WIN:
                self.N.add(self.encodeBoard(game))

            if b_first_result == PROVEN_LOSS and w_first_result == PROVEN_LOSS:
                self.P[self.encodeBoard(game)] = max_depth


eg = EndGameDB()

MAX_LENGTH = 18

for i in range(2, MAX_LENGTH):
    eg.generateGameCombinations("", 0, i)

eg.createLRclasses()
eg.createNPpositions()

print(eg.games)


with open('winning_b.txt', 'w') as file:
    file.write(str(eg.WINING_B))

with open('winning_w.txt', 'w') as file:
    file.write(str(eg.WINING_W))


with open('p.txt', 'w') as file:
    file.write(str(eg.P))

with open('n.txt', 'w') as file:
    file.write(json.dumps(list(eg.N)))

