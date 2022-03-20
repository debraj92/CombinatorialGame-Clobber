import unittest
import time

from boolean_negamax_tt import PlayClobber
from clobber_1d import Clobber_1d
from game_basics import EMPTY, BLACK, WHITE, isEmptyBlackWhite, opponent
import keras.models
import numpy as np
import tensorflow as tf


class clobberInstanceTests(unittest.TestCase):
    def test1(self):
        clobber = Clobber_1d("BWBW", BLACK)
        h1 = clobber.getBoardHash()
        clobber.play((0, 1))
        h2 = clobber.getBoardHash()
        self.assertNotEqual(h1, h2)
        clobber.undoMove((0, 1), h1)
        h3 = clobber.getBoardHash()
        self.assertEqual(h1, h3)

    def test2(self):
        clobber = Clobber_1d("BWBW", BLACK)
        moves = []
        legalmoves = clobber.computePrunedMovesFromSubgames(0)
        for move_set, _, _, _, _, in legalmoves:
            for nextMove in move_set:
                moves.append(nextMove)

        self.assertTrue(moves[0] == (0, 1)) #black wins
        self.assertTrue((2, 1) in moves)
        self.assertTrue((2, 3) in moves)
        self.assertEqual(3, len(moves))

    def test2_(self):
        clobber = Clobber_1d("BWBW", BLACK)
        clobber.play((0, 1))
        clobber.play((3, 2))
        moves = []
        legalmoves = clobber.computePrunedMovesFromSubgames(0)
        for move_set, _, _, _, _, in legalmoves:
            for nextMove in move_set:
                moves.append(nextMove)

        self.assertEqual(len(moves), 1)
        self.assertTrue((1, 2) in moves)

    def test4(self):
        clobber = Clobber_1d("BBW", BLACK, 1)
        clobber.updateBoardHashValue((1, 2))
        h1 = clobber.getBoardHash()
        clobber = Clobber_1d("B.B", WHITE, 1)
        h2 = clobber.getBoardHash()
        self.assertEqual(h1, h2)

    def test5(self):
        clobber = Clobber_1d("BBW", BLACK, 1)
        h1 = clobber.getBoardHash()
        clobber.updateBoardHashValue((1, 2))
        h2 = clobber.getBoardHash()
        self.assertNotEqual(h1, h2)

    def test6(self):
        clobber = Clobber_1d("BBWW", WHITE, 1)
        clobber.updateBoardHashValue((2, 1))
        h1 = clobber.getBoardHash()
        clobber = Clobber_1d("BWWW", BLACK, 1)
        h2 = clobber.getBoardHash()
        self.assertNotEqual(h1, h2)

    def test7(self):
        clobber = Clobber_1d("BBWW", WHITE, 1)
        clobber.updateBoardHashValue((2, 1))
        h1 = clobber.getBoardHash()
        clobber = Clobber_1d("BW.W", BLACK, 1)
        h2 = clobber.getBoardHash()
        self.assertEqual(h1, h2)

    def test9(self):
        clobber = Clobber_1d("BWBWWB.WBWBBW", WHITE)
        moves = clobber.computePrunedMovesFromSubgames(True)
        assert len(moves) == 0

    def test12(self):
        clobber = Clobber_1d("BBWW", BLACK, 1)
        clobber.computePrunedMovesFromSubgames(True)
        clobber.applyMoveForFeatureEvaluation((1, 2))
        boardString = ""
        for i in range(len(clobber.board_features)):
            if clobber.board_features[i][0] == 0 and clobber.board_features[i][1] == 1:
                boardString += "B"
            elif clobber.board_features[i][0] == 1 and clobber.board_features[i][1] == 0:
                boardString += "W"
            else:
                boardString += "."

        assert boardString == "B.BW...................................."

    def test13(self):
        clobber = Clobber_1d("BWBW", WHITE, 1)
        clobber.computePrunedMovesFromSubgames(True)
        clobber.applyMoveForFeatureEvaluation((1, 0))
        boardString = ""
        for i in range(len(clobber.board_features)):
            if clobber.board_features[i][0] == 0 and clobber.board_features[i][1] == 1:
                boardString += "B"
            elif clobber.board_features[i][0] == 1 and clobber.board_features[i][1] == 0:
                boardString += "W"
            else:
                boardString += "."

        assert boardString == "W.BW...................................."

    def testBlackModel_1(self):
        model_black = keras.models.load_model('clobber-black-cnn.h5')
        clobber = Clobber_1d("BWB", BLACK, 1)  # Exp : Black wins(1) [0 1]
                                                # we are evaluating from black's perspective.
        clobber.computePrunedMovesFromSubgames(True)
        X = clobber.board_features
        X = np.reshape(X, (1, 40, 2))
        prediction = model_black.predict(X)
        print(prediction)
        assert prediction[0][1] > 0.7
        assert prediction[0][0] < 0.3

    def testBlackModel_2(self):
        model_black = keras.models.load_model('clobber-black-cnn.h5')
        clobber = Clobber_1d("BBWW", BLACK, 1)  # Exp : Black loses(0) [1 0], ignore player -
                                                # we are evaluating from black's perspective.
        clobber.computePrunedMovesFromSubgames(True)
        X = clobber.board_features
        X = np.reshape(X, (1, 40, 2))
        prediction = model_black.predict(X)
        assert prediction[0][1] < 0.3
        assert prediction[0][0] > 0.7

    def testWhiteModel_1(self):
        model_black = keras.models.load_model('clobber-white-cnn.h5')
        clobber = Clobber_1d("WB", BLACK, 1)  # Exp : White wins(1) [0 1]
        clobber.computePrunedMovesFromSubgames(True)
        X = clobber.board_features
        X = np.reshape(X, (1, 40, 2))
        prediction = model_black.predict(X)
        assert prediction[0][1] > 0.7
        assert prediction[0][0] < 0.3

    def testWhiteModel_2(self):
        model_black = keras.models.load_model('clobber-white-cnn.h5')
        clobber = Clobber_1d("BBWW", BLACK, 1)  # Exp : White loses(0) [1 0]
        clobber.computePrunedMovesFromSubgames(True)
        X = clobber.board_features
        X = np.reshape(X, (1, 40, 2))
        prediction = model_black.predict(X)
        assert prediction[0][1] < 0.3
        assert prediction[0][0] > 0.7

    def testchecktf(self):
        label = np.array([1, 0, 1, 1, 0])
        label = tf.keras.utils.to_categorical(label, num_classes=2)
        print(label)
        print(label.shape)

    def testprediction(self):
        model_black = keras.models.load_model('./clobber-black-cnn.h5')
        clobber = Clobber_1d("BBW", BLACK, 1)  # Exp : White loses(0) [1 0]
        clobber.computePrunedMovesFromSubgames(True)
        X = clobber.board_features
        start = time.time()
        X = np.reshape(X, (1, 40, 2))
        prediction = model_black.predict(X)
        print(prediction)
        end = time.time()
        d = end - start
        print("time taken ", d)
