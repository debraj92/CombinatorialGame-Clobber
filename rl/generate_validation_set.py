# Generate all possible boards of size N
# Play board twice - get outcome class
# Store
# {board: 00/01/10/11}} BLACK-WHITE: 1=First player win
# Eval: Generate K boards. Calculate number of winning games. Play via RL. Calculate winning games. Compare.

import os
import sys
import pickle
import time
import itertools
from tqdm import tqdm

sys.path.append("./")
from game_basics import EMPTY, BLACK, WHITE
from boolean_negamax_tt import PlayClobber
from clobber_1d import Clobber_1d
from search_basics import PROVEN_WIN, PROVEN_LOSS


if __name__ == "__main__":
    MAX_BOARD_SIZE = 15
    OUTPUT_FILE = os.path.join("rl", "validation_set.pkl")

    database = {}
    board_converter = {"B": BLACK, "W": WHITE, ".": EMPTY}

    for board_length in range(MAX_BOARD_SIZE + 1):
        for board in tqdm(list(itertools.product(["B", "W", "."], repeat=board_length))):
            clobber = Clobber_1d(board, BLACK)
            play = PlayClobber()
            black_first = play.negamaxClobberGamePlay(clobber, time.time())[0]

            clobber = Clobber_1d(board, WHITE)
            play = PlayClobber()
            white_first = play.negamaxClobberGamePlay(clobber, time.time())[0]

            if black_first == PROVEN_WIN and white_first == PROVEN_WIN:
                outcome = "11"
            elif black_first == PROVEN_WIN and white_first == PROVEN_LOSS:
                outcome = "10"
            elif black_first == PROVEN_LOSS and white_first == PROVEN_WIN:
                outcome = "01"
            elif black_first == PROVEN_LOSS and white_first == PROVEN_LOSS:
                outcome = "00"

            # Convert board to the format we want I.E. using BLACK/WHITE/EMPTY
            board = tuple([board_converter[token] for token in board])

            database[board] = outcome
        print(f"Board length {board_length} complete.")

    with open(OUTPUT_FILE, "wb") as fp:
        pickle.dump(database, fp)
