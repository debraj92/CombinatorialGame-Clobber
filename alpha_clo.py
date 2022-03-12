#!/usr/bin/env python3
from clobber_1d import Clobber_1d
from boolean_negamax_tt import PlayClobber
from game_basics import EMPTY, BLACK, WHITE
from search_basics import INFINITY, PROVEN_WIN, PROVEN_LOSS, UNKNOWN
import time
import sys


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


def string_to_player(string):
    if string == "B":
        return BLACK
    elif string == "W":
        return WHITE


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


if __name__ == "__main__":
    # Inputs
    board = sys.argv[1]
    first_player = sys.argv[2]
    timeout = float(sys.argv[3])
    # Start timer
    start = time.time()
    # Setup
    first_player = string_to_player(first_player)
    clobber = Clobber_1d(board, first_player)
    # Run algorithm
    play = PlayClobber()
    outcome, winning_move, nodes = play.negamaxClobberGamePlay(
        clobber, start, timeout=timeout
    )
    # Stop timer
    end = time.time()
    # Print results
    print(
        f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
    )
