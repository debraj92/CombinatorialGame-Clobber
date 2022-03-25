from clobber_1d import Clobber_1d
from boolean_negamax_tt import PlayClobber
import time
from game_basics import EMPTY, BLACK, WHITE

import unittest

PROVEN_WIN = 10000
PROVEN_LOSS = -10000
UNKNOWN = -5
PROVEN = 0
DISPROVEN = 1
INFINITY = 1000000


play = PlayClobber()
totalPlayer = BLACK + WHITE


class clobberCNN_AccuracyTests(unittest.TestCase):

    def test1(self):
        game = "BWBWBWBWBWBW"
        first_player = BLACK
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        if outcome1 == PROVEN_WIN:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            clobber2.play(winning_move)
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())
        else:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())

        self.assertTrue(outcome2 == PROVEN_LOSS)

    def test2(self):
        game = "WBWBWBWBWBWBWBWBW"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        if outcome1 == PROVEN_WIN:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            clobber2.play(winning_move)
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())
        else:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())

        self.assertTrue(outcome2 == PROVEN_LOSS)

    def test3(self):
        game = "WBWBWBBWBWBWBWBBBWBWWB..WBWBW"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        if outcome1 == PROVEN_WIN:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            clobber2.play(winning_move)
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())
        else:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())

        self.assertTrue(outcome2 == PROVEN_LOSS)

    def test4(self):
        game = "BBW..BBW"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        if outcome1 == PROVEN_WIN:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            clobber2.play(winning_move)
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())
        else:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())

        self.assertTrue(outcome2 == PROVEN_LOSS)

    def test5(self):
        game = "BB..B"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        if outcome1 == PROVEN_WIN:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            clobber2.play(winning_move)
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())
        else:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())

        self.assertTrue(outcome2 == PROVEN_LOSS)

    def test6(self):
        game = "BBWBBWWWBWWBWWBBBW"
        first_player = BLACK
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        if outcome1 == PROVEN_WIN:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            clobber2.play(winning_move)
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())
        else:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())

        self.assertTrue(outcome2 == PROVEN_LOSS)

    def test7(self):
        game = "WBWBWBWBWBWBWBWBWBWBWBWBWBWB"
        first_player = BLACK
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        if outcome1 == PROVEN_WIN:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            clobber2.play(winning_move)
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())
        else:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())

        self.assertTrue(outcome2 == PROVEN_LOSS)

    def test8(self):
        game = "WBWBWBWBWBWBWBWBWBWBWBWBWBWB"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        if outcome1 == PROVEN_WIN:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            clobber2.play(winning_move)
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())
        else:
            clobber2 = Clobber_1d(game, first_player)
            clobber2.cnn_enabled = False
            outcome2, winning_move, nodes = play.negamaxClobberGamePlay(clobber2, time.time())

        self.assertTrue(outcome2 == PROVEN_LOSS)


if __name__ == "__main__":
    unittest.main()
