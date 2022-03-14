import unittest

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
        moves = clobber.computePrunedMovesFromSubgames()[0][0]
        self.assertTrue((0, 1) in moves)
        self.assertTrue((2, 1) in moves)
        self.assertTrue((2, 3) in moves)
        self.assertEqual(3, len(moves))

    def test2_(self):
        clobber = Clobber_1d("BWBW", BLACK)
        clobber.play((0, 1))
        clobber.play((3, 2))
        moves = clobber.computePrunedMovesFromSubgames()[0][0]
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
        moves = clobber.computePrunedMovesFromSubgames()
        assert len(moves) == 0

    def test10(self):
        clobber = Clobber_1d("BWBWBWBW.WBWBWBB", BLACK, 1)
        play = PlayClobber()
        value = play.evaluateMove(clobber, (0, 1), play.model_white)
        assert value == -1

        clobber = Clobber_1d("BBWW", BLACK, 1)
        play = PlayClobber()
        value = play.evaluateMove(clobber, (1, 2), play.model_white)
        assert value == 1

    def test11(self):
        clobber = Clobber_1d("BWBWBWBWBWBWBWBWBW", BLACK, 1)
        play = PlayClobber()
        value = play.evaluateMove(clobber, (0, 1), play.model_white)
        assert value == -1

        clobber = Clobber_1d("BWBWBW", BLACK, 1)
        play = PlayClobber()
        value = play.evaluateMove(clobber, (0, 1), play.model_white)
        assert value == 1

    def test12(self):
        clobber = Clobber_1d("BBWW", BLACK, 1)
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
        X = clobber.board_features
        X = np.reshape(X, (1, 40, 2))
        prediction = model_black.predict(X)
        assert prediction[0][1] > 0.8
        assert prediction[0][0] < 0.2

    def testBlackModel_2(self):
        model_black = keras.models.load_model('clobber-black-cnn.h5')
        clobber = Clobber_1d("BBWW", BLACK, 1)  # Exp : Black loses(0) [1 0], ignore player -
                                                # we are evaluating from black's perspective.
        X = clobber.board_features
        X = np.reshape(X, (1, 40, 2))
        prediction = model_black.predict(X)
        assert prediction[0][1] < 0.2
        assert prediction[0][0] > 0.8

    def testWhiteModel_1(self):
        model_black = keras.models.load_model('clobber-white-cnn.h5')
        clobber = Clobber_1d("WB", BLACK, 1)  # Exp : White wins(1) [0 1]
        X = clobber.board_features
        X = np.reshape(X, (1, 40, 2))
        prediction = model_black.predict(X)
        assert prediction[0][1] > 0.8
        assert prediction[0][0] < 0.2

    def testWhiteModel_2(self):
        model_black = keras.models.load_model('clobber-white-cnn.h5')
        clobber = Clobber_1d("BBWW", BLACK, 1)  # Exp : White loses(0) [1 0]
        X = clobber.board_features
        X = np.reshape(X, (1, 40, 2))
        prediction = model_black.predict(X)
        assert prediction[0][1] < 0.2
        assert prediction[0][0] > 0.8

    def testchecktf(self):
        label = np.array([1, 0, 1, 1, 0])
        label = tf.keras.utils.to_categorical(label, num_classes=2)
        print(label)
        print(label.shape)
