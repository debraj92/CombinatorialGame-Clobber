import unittest
from clobber_1d import Clobber_1d
from game_basics import EMPTY, BLACK, WHITE, isEmptyBlackWhite, opponent


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
        moves = clobber.computeLegalMoves()
        self.assertTrue((0, 1) in moves)
        self.assertTrue((2, 1) in moves)
        self.assertTrue((2, 3) in moves)
        self.assertEqual(3, len(moves))

    def test2_(self):
        clobber = Clobber_1d("BWBW", BLACK)
        clobber.play((0, 1))
        clobber.play((3, 2))
        moves = clobber.computeLegalMoves()
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

