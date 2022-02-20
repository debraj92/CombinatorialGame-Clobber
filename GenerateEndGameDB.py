import json
import time

from game_basics import EMPTY, BLACK, WHITE
from boolean_negamax_tt import PlayClobber
from clobber_1d import Clobber_1d
from search_basics import INFINITY, PROVEN_WIN, PROVEN_LOSS, UNKNOWN


class EndGameDB:
    players = ["B", "W"]

    games = set()

    WINING_B = set()
    WINING_W = set()

    def generateGameCombinations(self, boardString, length, maxlength):

        if length == maxlength:
            self.games.add(boardString)
            return

        for i in self.players:
            self.generateGameCombinations(boardString + i, length + 1, maxlength)

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
                    game = game.replace("B", "1")
                    game = game.replace("W", "2")
                    self.WINING_B.add(game)
            else:

                first_player = WHITE
                clobber = Clobber_1d(game, first_player)
                play = PlayClobber()
                start_time = time.time()
                w_first_result, _, _ = play.negamaxClobberGamePlay(clobber, start_time)

                if w_first_result == PROVEN_WIN:
                    game = game.replace("B", "1")
                    game = game.replace("W", "2")
                    self.WINING_W.add(game)


eg = EndGameDB()

MAX_LENGTH = 13

for i in range(2, MAX_LENGTH):
    eg.generateGameCombinations("", 0, i)

eg.createLRclasses()

print(eg.games)
#print("WINNING B ", eg.WINING_B)
#print("WINNING W ", eg.WINING_W)

with open('winning_b.txt', 'w') as file:
    file.write(json.dumps(list(eg.WINING_B)))

with open('winning_w.txt', 'w') as file:
    file.write(json.dumps(list(eg.WINING_W)))
