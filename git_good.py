#!/usr/bin/env python3
from clobber_1d import Clobber_1d
from boolean_negamax_tt import PlayClobber
from game_basics import EMPTY, BLACK, WHITE
from search_basics import INFINITY, PROVEN_WIN, PROVEN_LOSS, UNKNOWN
import time
import sys
import argparse


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
    parser = argparse.ArgumentParser(description="Solve a clobber board.")
    parser.add_argument(
        "board",
        type=str,
        help="Clobber board to solve. Must contain only B, W or .",
    )
    parser.add_argument(
        "player",
        type=str,
        help="First player. Must be B or W",
    )
    parser.add_argument(
        "timeout",
        type=float,
        help="Maximum time allowed. Must be a number.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--default_move_ordering",
        action='store_true',
        default=False,
        help="Uses the default move ordering scheme.",
    )
    group.add_argument(
        "--cnn_move_ordering",
        action='store_true',
        default=False,
        help="Uses the CNN move ordering scheme.",
    )
    group.add_argument(
        "--rl_move_ordering",
        action='store_true',
        default=False,
        help="Uses the RL move ordering scheme.",
    )
    group.add_argument(
        "--no_move_ordering",
        action='store_true',
        default=False,
        help="Uses no move (I.E. random) ordering scheme.",
    )

    args = parser.parse_args()

    board = args.board
    first_player = args.player
    timeout = args.timeout
    move_ordering = {
        "default": args.default_move_ordering,
        "cnn": args.cnn_move_ordering,
        "rl": args.rl_move_ordering,
        "none": args.no_move_ordering
    }
    # Start timer
    start = time.time()
    # Setup
    first_player = string_to_player(first_player)
    clobber = Clobber_1d(board, first_player)
    clobber.cnn_enabled = True if args.cnn_move_ordering else False
    # Run algorithm
    play = PlayClobber(move_ordering=move_ordering)
    outcome, winning_move, nodes = play.negamaxClobberGamePlay(
        clobber, start, timeout=timeout
    )
    # Stop timer
    end = time.time()
    # Print results
    print(
        f"{outcome_to_string(outcome, first_player)} {format_winning_move(winning_move)} {end - start} {nodes}"
    )
