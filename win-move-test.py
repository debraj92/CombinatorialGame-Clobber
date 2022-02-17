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

def opponent(player):
    if player == BLACK:
        return WHITE
    else:
        return BLACK


def player_to_string(player):
    if player == BLACK:
        return "B"
    elif player == WHITE:
        return "W"


def outcome_to_string(outcome, first_player):
    if outcome == UNKNOWN:
        return "?"
    elif outcome == PROVEN_WIN:
        return player_to_string(first_player)
    else:
        return player_to_string(opponent(first_player))


def format_winning_move(winning_move):
    if winning_move:
        src, target = winning_move
        return f"{src}-{target}"
    else:
        return "None"

play = PlayClobber()

class clobberPlayTests(unittest.TestCase):

    def test1(self):
        start = time.time()
        first_player = BLACK
        W = WHITE
        B = BLACK
        clobber = Clobber_1d("WBBWB", B)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )

        if outcome == PROVEN_WIN:
            print("Win")
        else:
            print("lose")

        print(winning_move)
        assert winning_move == (4, 3)


    def test2(self):
        start = time.time()
        first_player = BLACK
        W = WHITE
        B = BLACK
        clobber = Clobber_1d("WB", B)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )

        assert winning_move == (1,0)

    def test3(self):
        start = time.time()
        first_player = BLACK
        W = WHITE
        B = BLACK
        clobber = Clobber_1d("WBWB", W)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        print(winning_move)
        assert winning_move == (0,1)

    def test4(self):
        start = time.time()
        first_player = BLACK
        W = WHITE
        B = BLACK
        clobber = Clobber_1d("BBWBWB", B)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        print(winning_move)
        assert winning_move == (5,4)

    def test5(self):
        start = time.time()
        first_player = BLACK
        W = WHITE
        B = BLACK
        clobber = Clobber_1d("BWBWBWWB", W)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        print(winning_move)
        assert winning_move == (1,2)

if __name__ == "__main__":
    unittest.main()
