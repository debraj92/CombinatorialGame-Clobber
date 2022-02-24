from boolean_negamax_tt import PlayClobber
from clobber_1d import Clobber_1d
import time
from game_basics import EMPTY, BLACK, WHITE
from search_basics import INFINITY, PROVEN_WIN, PROVEN_LOSS, UNKNOWN

import unittest


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

class TimeoutTest(unittest.TestCase):
    def test1(self):
        print("\nTest 1")
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 1
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test2(self):
        print("\nTest 2")
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 1
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test3(self):
        print("\nTest 3")
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 2
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test4(self):
        print("\nTest 4")
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 1.5
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, UNKNOWN)

    def test5(self):
        print("\nTest 5")
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWB.WBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 1
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test6(self):
        print("\nTest 6")
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 4
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test7(self):
        print("\nTest 7")
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 11
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, UNKNOWN)

    def test8(self):
        print("\nTest 8")
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 10
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, UNKNOWN)

    def test9(self):
        print("\nTest 9")
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 1
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, UNKNOWN)

    def test10(self):
        print("\nTest 10")
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 4
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test11(self):
        print("\nTest 11")
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 30
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)


if __name__ == "__main__":
    unittest.main()
