from boolean_negamax_tt import PlayClobber
from clobber_1d import Clobber_1d
import time
from search_basics import INFINITY, PROVEN_WIN, PROVEN_LOSS, UNKNOWN
from game_basics import EMPTY, BLACK, WHITE

# Basic tests
start = time.time()
print("Test 1")
test_start = time.time()
first_player = WHITE
clobber = Clobber_1d("BBB.WB", first_player)
play = PlayClobber()
clobber.print_()
outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, test_start)
assert outcome == PROVEN_WIN
print()

print("Test 2")
test_start = time.time()
first_player = WHITE
clobber = Clobber_1d("BBWBB", first_player)
play = PlayClobber()
clobber.print_()
outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, test_start)
assert outcome == PROVEN_LOSS
print()

print("Test 3")
test_start = time.time()
first_player = WHITE
clobber = Clobber_1d("BBWBBWWBB", first_player)
clobber.print_()
play = PlayClobber()
outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, test_start)
assert outcome == PROVEN_LOSS
print()

print("Test 4")
test_start = time.time()
first_player = WHITE
clobber = Clobber_1d("BW.WB.WW.WBB", first_player)
clobber.print_()
play = PlayClobber()
outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, test_start)
assert outcome == PROVEN_LOSS
print()

print("Test 6")
test_start = time.time()
first_player = WHITE
clobber = Clobber_1d("..WWW..BWBW.BWBB.WBWBWW...", first_player)
clobber.print_()
play = PlayClobber()
outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, test_start)
assert outcome == PROVEN_WIN
print()

print("Test 7")
test_start = time.time()
first_player = WHITE
clobber = Clobber_1d("BWBWBWBWBWBWBWBWBWB", first_player)
clobber.print_()
play = PlayClobber()
outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, test_start)
assert outcome == PROVEN_WIN
print()

print("Test 8")
test_start = time.time()
first_player = WHITE
clobber = Clobber_1d("BBWBWBWB", first_player)
clobber.print_()
play = PlayClobber()
outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, test_start)
assert outcome == PROVEN_WIN
print()

print("Test 9")
test_start = time.time()
first_player = WHITE
clobber = Clobber_1d("BBWBWBWBWBWB..WWW..BWBW.BWBB", first_player)
clobber.print_()
play = PlayClobber()
outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, test_start)
assert outcome == PROVEN_WIN
print()

print("Test 10")
test_start = time.time()
first_player = WHITE
clobber = Clobber_1d("WBBWBWWBB.WBBWBBWB..WBWBB", first_player)
clobber.print_()
play = PlayClobber()
outcome, winning_move, nodes = play.negamaxClobberGamePlay(clobber, test_start)
assert outcome == PROVEN_LOSS

end = time.time()
print(end - start)
"""
Analysis (8):
BBWBWBWB

White Moves: (4,5)
BBWB.WWB

Black Moves: (7,6) and (3,2)

(7,6)
BBWB.WB.

White Move: (2,1)
BW.B.WB.

Black Moves: (0,1) and (6,5)

(0,1) 
.B.B.WB. ==> WHITE's TURN AND WINS 
 
(6,5)
BW.B.B.. ==> WHITE's TURN AND WINS

BBWB.WWB
(3,2) ==> BLACK's TURN

BBB..WWB ==> WHITE's TURN AND WINS

"""
