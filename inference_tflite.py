import time

import tensorflow as tf
import numpy as np
from random import choices

from boolean_negamax_tt import PlayClobber
from clobber_1d import Clobber_1d
import sys

games_for_training = []

PROVEN_WIN = 10000
PROVEN_LOSS = -10000

TEST_BLACK = True
TEST_WHITE = True

move_ordering = {"rl": False, "cnn": True, "default": False, "none": False}

def createRandomBoard(size, skipDot):
    if skipDot:
        elements = ["B", "W"]
    else:
        elements = [".", "B", "W"]#["B", "W", "."]
    board = ""
    population = [0, 1, 2]
    weights = [0.4, 0.3, 0.3]
    #weights = [0.65, 0.18, 0.17]
    for i in range(size):
        index = choices(population, weights)
        #board += elements[np.random.randint(low=0, high=3)]

        board += elements[index[0]]

    return board


def generateGameCombinations(sample_size, boardSize, skipDots=False):
    global games_for_training
    for i in range(sample_size):
        games_for_training.append(createRandomBoard(boardSize, skipDots))


def repeatGame(game, count):
    global games_for_training
    for i in range(count):
        games_for_training.append(game)


def printGames():
    global games_for_training
    for i in range(len(games_for_training)):
        print(games_for_training[i])


def dotifyGame(game, boardSize, count_of_positons_to_delete):
    dots = np.random.choice(boardSize - 1, count_of_positons_to_delete, replace=False)
    for d in dots:
        game = game[:d] + "." + game[d + 1:]

    return game


def dotifyGames(boardSize, count_of_positons_to_delete):
    global games_for_training
    for i in range(len(games_for_training)):
        game = games_for_training[i]
        games_for_training[i] = dotifyGame(game, boardSize, count_of_positons_to_delete)


def compress():
    global games_for_training
    compressed_games = []
    for i in range(len(games_for_training)):
        newGame = ""
        game = games_for_training[i]
        previous_char = None
        for p in game:
            if previous_char is not None and previous_char == "." and p == ".":
                continue
            else:
                newGame += p
                previous_char = p

        dots_to_be_added = len(game) - len(newGame)
        for i in range(dots_to_be_added):
            newGame += "."

        compressed_games.append(newGame)
    return compressed_games


'''
board = "BWBWBWBWBWBWBWBWBWBWBWBWBWBWBWBWBWBWBWBW"
repeatGame(board, 5000)
boardSize = len(board)
depth = 14
count_of_positions_to_delete = boardSize - depth
dotifyGames(boardSize, count_of_positions_to_delete)
printGames()
'''

boardSize = 35
generateGameCombinations(10000, boardSize)
printGames()
#print()
#games_for_training = compress()
#printGames()
#sys.exit()


def inference_black(board):
    clobber = Clobber_1d(board, 1)  # Exp : White loses(0) [1 0]
    clobber.computePrunedMovesFromSubgames(True)
    X = clobber.board_features
    X = np.reshape(X, (1, 40, 2))

    start = time.time()
    # Load TFLite model and allocate tensors.
    #interpreter = tf.lite.Interpreter(model_path="./final-models/m7_b_m9_w/clobber-black-cnn.tflite")
    interpreter = tf.lite.Interpreter(model_path="./cnn-models/best3/clobber-black-cnn.tflite")
    #interpreter = tf.lite.Interpreter(model_path="./clobber-black-cnn.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    d = end - start
    # print("time taken ", d)
    # print(output_data)
    return output_data


if TEST_BLACK:

    count_correct = 0
    count_incorrect = 0
    exec_time = 0
    print("Checking accuracy for Black")
    for game in games_for_training:
        #print(game)
        clobber = Clobber_1d(game, 1)
        play = PlayClobber(move_ordering)
        test_start = time.time()
        expected_outcome, _, _ = play.negamaxClobberGamePlay(clobber, test_start, 1000)
        start = time.time()
        #print(expected_outcome)
        predicted_outcome = inference_black(game)
        #print(predicted_outcome)
        end = time.time()
        exec_time += end - start
        if expected_outcome == PROVEN_WIN and predicted_outcome[0][1] > 0.7 and predicted_outcome[0][0] < 0.3:
            count_correct += 1
        elif expected_outcome == PROVEN_LOSS and predicted_outcome[0][1] < 0.3 and predicted_outcome[0][0] > 0.7:
            count_correct += 1
        else:
            count_incorrect += 1


    test_accuracy = 100 * (count_correct / (count_correct + count_incorrect))
    print("Test Accuracy Black ", test_accuracy)
    avg_exec_time = exec_time / len(games_for_training)
    print("Average execution time ", avg_exec_time)


# white


def inference_white(board):
    # print(board)
    clobber = Clobber_1d(board, 1)  # Exp : White loses(0) [1 0]
    clobber.computePrunedMovesFromSubgames(True)
    X = clobber.board_features
    X = np.reshape(X, (1, 40, 2))

    start = time.time()
    # Load TFLite model and allocate tensors.
    #interpreter = tf.lite.Interpreter(model_path="./final-models/m7_b_m9_w/clobber-white-cnn.tflite")
    interpreter = tf.lite.Interpreter(model_path="./cnn-models/best3/clobber-white-cnn.tflite")
    #interpreter = tf.lite.Interpreter(model_path="./clobber-white-cnn.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    d = end - start
    # print("time taken ", d)
    # print(output_data)
    return output_data


if TEST_WHITE:
    # print(inference_white2("B.B.BBWW.W.BW.BW"))

    count_correct = 0
    count_incorrect = 0
    exec_time = 0
    print("Checking accuracy for White")
    for game in games_for_training:
        #print(game)
        clobber = Clobber_1d(game, 2)
        play = PlayClobber(move_ordering)
        test_start = time.time()
        expected_outcome, _, _ = play.negamaxClobberGamePlay(clobber, test_start, 1000)

        start = time.time()
        # print(expected_outcome)
        predicted_outcome = inference_white(game)
        # print(predicted_outcome)
        end = time.time()
        exec_time += end - start
        if predicted_outcome is None:
            continue
        if expected_outcome == PROVEN_WIN and predicted_outcome[0][1] > 0.8 and predicted_outcome[0][0] < 0.2:
            count_correct += 1
        elif expected_outcome == PROVEN_LOSS and predicted_outcome[0][1] < 0.2 and predicted_outcome[0][0] > 0.8:
            count_correct += 1
        else:
            count_incorrect += 1

    test_accuracy = 100 * (count_correct / (count_correct + count_incorrect))
    print("Test Accuracy White ", test_accuracy)
    avg_exec_time = exec_time / len(games_for_training)
    print("Average execution time ", avg_exec_time)
