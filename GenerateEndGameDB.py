import json
import time

#from a2.boolean_negamax_tt import PlayClobber
#from a2.clobber_1d import Clobber_1d
from boolean_negamax_tt import PlayClobber
from clobber_1d import Clobber_1d
from game_basics import EMPTY, BLACK, WHITE
from search_basics import INFINITY, PROVEN_WIN, PROVEN_LOSS, UNKNOWN


class EndGameDB:
    players = ["B", "W"]

    elements = ["B", "W", "."]
    elements_integers = ["1", "2", "0"]

    games = []#set()
    games_map = dict()

    N = set()
    P = dict()
    WINING_B = dict()
    WINING_W = dict()

    endgame_B = dict()
    endgame_W = dict()

    def generateGameCombinations(self, boardString, length, maxlength):

        if length == maxlength:
            self.games.append(boardString)
            return

        for i in self.players:
            self.generateGameCombinations(boardString + i, length + 1, maxlength)

    def generateAllGameCombinations(self, boardString, boardString_int, length, maxlength):

        if length == maxlength:
            self.games.append(boardString)
            return

        for i, p in enumerate(self.elements):
            self.generateAllGameCombinations(boardString + p, boardString_int + self.elements_integers[i], length + 1, maxlength)

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

    def compressGames(self):

        convertMap = {".": "0", "B": "1", "W": "2"}
        for i, game in enumerate(self.games):

            current = convertMap[game[0]]
            current_raw = game[0]
            j = 1
            new_game_str = current
            new_game_str_raw = current_raw
            while j < len(game):
                if current == "0" and game[j] == ".":
                    j += 1
                    continue
                else:
                    current = convertMap[game[j]]
                    new_game_str += current
                    new_game_str_raw += game[j]
                    j += 1

            self.games_map[new_game_str] = new_game_str_raw

    def printGames(self):

        for game_str, game_str_raw in self.games_map.items():
            print(game_str +", ",game_str_raw)

    def createEndGameDB(self):
        for game_str, game_str_raw in self.games_map.items():
            first_player = BLACK
            clobber = Clobber_1d(game_str_raw, first_player)
            play = PlayClobber()
            start_time = time.time()
            outcome, _, _ = play.negamaxClobberGamePlay(clobber, start_time)
            print(game_str_raw, ", " + str(outcome))
            outcome /= PROVEN_WIN
            self.endgame_B[game_str] = outcome

            first_player = WHITE
            clobber = Clobber_1d(game_str_raw, first_player)
            play = PlayClobber()
            start_time = time.time()
            outcome, _, _ = play.negamaxClobberGamePlay(clobber, start_time)
            print(game_str_raw, ", " + str(outcome))
            outcome /= PROVEN_WIN
            self.endgame_W[game_str] = outcome

        print("Saving to disk")
        with open('endgame_b.txt', 'w') as file:
            file.write(str(self.endgame_B))

        with open('endgame_w.txt', 'w') as file:
            file.write(str(self.endgame_W))


eg = EndGameDB()

MAX_LENGTH = 12
print("Generating all combination of size ", MAX_LENGTH)
eg.generateAllGameCombinations("", "", 0, MAX_LENGTH)
eg.compressGames()
#eg.printGames()
print("Start Playing")
eg.createEndGameDB()

'''
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
'''
