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

totalPlayer = BLACK + WHITE


class clobberCNN_AccuracyTests(unittest.TestCase):

    def test1(self):
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": False,
            "none": False
        }
        play = PlayClobber(move_ordering)
        game = "BWBWBWBWBWBW"
        first_player = BLACK
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True if move_ordering["cnn"] else False
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": True,
            "none": False
        }
        play = PlayClobber(move_ordering)
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
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": False,
            "none": False
        }
        play = PlayClobber(move_ordering)
        game = "WBWBWBWBWBWBWBWBW"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True if move_ordering["cnn"] else False
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": True,
            "none": False
        }
        play = PlayClobber(move_ordering)
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
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": False,
            "none": False
        }
        play = PlayClobber(move_ordering)
        game = "WBWBWBBWBWBWBWBBBWBWWB..WBWBW"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True if move_ordering["cnn"] else False
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": True,
            "none": False
        }
        play = PlayClobber(move_ordering)
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
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": False,
            "none": False
        }
        play = PlayClobber(move_ordering)
        game = "BBW..BBW"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True if move_ordering["cnn"] else False
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": True,
            "none": False
        }
        play = PlayClobber(move_ordering)
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
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": False,
            "none": False
        }
        play = PlayClobber(move_ordering)
        game = "BB..B"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True if move_ordering["cnn"] else False
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": True,
            "none": False
        }
        play = PlayClobber(move_ordering)
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
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": False,
            "none": False
        }
        play = PlayClobber(move_ordering)
        game = "BBWBBWWWBWWBWWBBBW"
        first_player = BLACK
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True if move_ordering["cnn"] else False
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": True,
            "none": False
        }
        play = PlayClobber(move_ordering)
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
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": False,
            "none": False
        }
        play = PlayClobber(move_ordering)
        game = "WBWBWBWBWBWBWBWBWBWBWBWBWBWB"
        first_player = BLACK
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True if move_ordering["cnn"] else False
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": True,
            "none": False
        }
        play = PlayClobber(move_ordering)
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
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": False,
            "none": False
        }
        play = PlayClobber(move_ordering)
        game = "WBWBWBWBWBWBWBWBWBWBWBWBWBWB"
        first_player = WHITE
        clobber = Clobber_1d(game, first_player)
        clobber.cnn_enabled = True if move_ordering["cnn"] else False
        outcome1, winning_move, nodes = play.negamaxClobberGamePlay(clobber, time.time())
        move_ordering = {
            "rl": False,
            "cnn": True,
            "default": True,
            "none": False
        }
        play = PlayClobber(move_ordering)
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
