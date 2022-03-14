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
        clobber = Clobber_1d("BW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test2(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test3(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test4(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBW", first_player)
        # outcome, winning_move, nodes = play.negamaxClobberIterativeDeepening(clobber, 1)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test5(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test6(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BBBW.BWW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test7(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BBBW.BWW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test8(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test9(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test10(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test11(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)
        # self.assertTrue(end - start < 1)

    def test12(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWB.WBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)
        self.assertTrue(end - start < 1)

    def test13(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWB.WBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)
        self.assertTrue(end - start < 1)

    def test14(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test15(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start, 5000)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test16(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWBW.WBWBWBB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test17(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBW.WBWBWBB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    """
    Analysis:
    WBWBWBWW
    Black: B.WBWBWW
    WHITE: B..WWBWW or B.WB.WWW
    Black: Wins
    """

    def test18(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("WBWBWBWW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test19(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("WWBWBWBWW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test20(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test21(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test22(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("WBWBWBWBWBWBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        # 4.21 - Black Wins, # 3.75 - WHITE WINS, Memory < 1 MB
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test23(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test24(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test24_2(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test25(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, start)
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    # BWBWBWBWBWBWBWBWBWBWBW (22) in 11s W  6s B
    # BWBWBWBWBWBWBWBWBWBWBWBW (24) in 13s W 38s B
    # BWBWBWBWBWBWBWBWBWBWBWBWBW (26) in 98s W only

    def test26(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test27(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test28(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("WBWBWBWBWBBBWBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 250
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test29(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("WBWBWBWBWBWBWBWWWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test30(self):
        start = time.time()
        first_player = WHITE
        clobber = Clobber_1d("WBBWWBWBWBBBWBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test31(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("WBWBWBWBWBWBWBWWWWBBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)

    def test32(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 300
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test33(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBW.BWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 300
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    def test34(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("BWBWBWBWBWBW.WBWBWBWBWBWB", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 300
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_LOSS)

    """
    Current:
    
    B 3-4 35.43396210670471 1706489
    
    B 3-4 49.849161863327026 2190776
    Previous:
    B 3-4 91.35586214065552 9069724
    
    Huge difference in the number of nodes searched (thanks to the 2nd player sum 0 pruning).
    """

    def test35(self):
        start = time.time()
        first_player = BLACK
        clobber = Clobber_1d("WBWBWBWBWBWBWBWBWBWBWBWBWBWBW", first_player)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print(
            f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
        )
        self.assertEqual(outcome, PROVEN_WIN)



    # 0.8 seconds
    # 24 positions
    def test36(self):
        start = time.time()
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBWBW", BLACK)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print("Total time ", end - start)
        self.assertEqual(outcome, PROVEN_WIN)



    #4.1s
    def test37(self):
        start = time.time()
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBWBBW", WHITE)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print("Total time ", end - start)
        self.assertEqual(outcome, PROVEN_WIN)



    # 0.9s
    def test38(self):
        start = time.time()
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWBWBWB", WHITE)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print("Total time ", end - start)
        self.assertEqual(outcome, PROVEN_WIN)

    def test39(self):
        start = time.time()
        clobber = Clobber_1d("BW.WBW.BBWW", BLACK)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print("Total time ", end - start)
        self.assertEqual(outcome, PROVEN_LOSS)

    def test40(self):
        start = time.time()
        clobber = Clobber_1d("W.B.WW.", BLACK)
        outcome, winning_move, nodes = play.negamaxClobberGamePlay(
            clobber, start, 150
        )
        end = time.time()
        print("Total time ", end - start)
        self.assertEqual(outcome, PROVEN_LOSS)

if __name__ == "__main__":
    unittest.main()
