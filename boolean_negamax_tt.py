import time
import keras.models
import numpy as np
import tensorflow as tf


class PlayClobber:
    # Transposition table to avoid re-computation of proven lost games
    proven_lost_states = set()

    # Transposition table to avoid re-computation of proven won games
    proven_win_states = set()

    moves_ = {}

    nodes_visited = set()

    out_of_time = False

    max_depth = 0

    def __init__(self):
        self.PROVEN_WIN = 10000
        self.PROVEN_LOSS = -10000
        self.UNKNOWN = -5
        self.PROVEN = 0
        self.DISPROVEN = 1
        self.INFINITY = 10000
        self.winningMove = ()
        #self.model_black_interpreter = self.modelInferenceInit("./final-models/m9/clobber-black-cnn.tflite")
        #self.model_white_interpreter = self.modelInferenceInit("./final-models/m9/clobber-white-cnn.tflite")
        self.model_black_interpreter = self.modelInferenceInit("./final-models/m7/clobber-black-cnn.tflite")
        self.model_white_interpreter = self.modelInferenceInit("./final-models/m9/clobber-white-cnn.tflite")

    def modelInferenceInit(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def blackModelInference(self, X):
        input_details = self.model_black_interpreter.get_input_details()
        output_details = self.model_black_interpreter.get_output_details()

        self.model_black_interpreter.set_tensor(input_details[0]['index'], X)
        self.model_black_interpreter.invoke()
        return self.model_black_interpreter.get_tensor(output_details[0]['index'])

    def whiteModelInference(self, X):
        input_details = self.model_white_interpreter.get_input_details()
        output_details = self.model_white_interpreter.get_output_details()

        self.model_white_interpreter.set_tensor(input_details[0]['index'], X)
        self.model_white_interpreter.invoke()
        return self.model_white_interpreter.get_tensor(output_details[0]['index'])

    def evaluateMove(self, state, move):
        state.applyMoveForFeatureEvaluation(move)
        X = state.board_features
        X = np.reshape(X, (1, 40, 2))
        if state.isCurrentPlayerWhite():
            prediction = self.blackModelInference(X)
        else:
            prediction = self.whiteModelInference(X)

        if prediction[0][0] < 0.3 and prediction[0][1] > 0.7:
            # current player wins
            sortKey = prediction[0][1]
        elif prediction[0][0] > 0.7 and prediction[0][1] < 0.3:
            # current player loses
            sortKey = -prediction[0][0]
        else:
            sortKey = 0

        state.undoMoveFromFeatureEvaluation(move)
        return sortKey

    def cnnMoveOrdering(self, state, legalMoves, previous_score, cnnOrdering):
        moves = []
        countWinMoves = 0
        noSort = previous_score <= -0.7
        for move_set, _, win, lose, _ in legalMoves:
            for nextMove in move_set:
                if cnnOrdering:
                    prediction = self.evaluateMove(state, nextMove)
                    if prediction > 0.7:
                        countWinMoves += 1
                        cnnOrdering = False
                    #if countWinMoves == 2:
                    #    cnnOrdering = False
                    moves.append((nextMove, prediction))
                else:

                    if win and not lose:
                        moves.append((nextMove, 0.6))
                    elif not win and lose:
                        moves.append((nextMove, -0.6))
                    else:
                        moves.append((nextMove, 1/len(move_set)))

        if previous_score > -0.7:
            moves = sorted(moves, key=lambda x: x[1])
        return moves

    def handleProcessingSubgames(self, legalMoves, boardHash):
        l_class = True
        r_class = True
        countN = 0
        countL = 0
        for move_set, _, win, lose, isN in legalMoves:
            l_class = l_class and win and not lose
            if l_class:
                countL += 1

            r_class = r_class and not win and lose

            if isN:
                countN += 1

        if len(legalMoves) > 0 and l_class:
            self.proven_win_states.add(boardHash)
            return self.INFINITY
        if len(legalMoves) > 0 and r_class:
            self.proven_lost_states.add(boardHash)
            return -self.INFINITY

        if len(legalMoves) == 1 and countN == 1:
            self.proven_win_states.add(boardHash)
            return self.INFINITY

        if countN == 1 and countL + countN == len(legalMoves):
            self.proven_win_states.add(boardHash)
            return self.INFINITY

        return 0

    def negamaxClobber1d(self, state, previous_score, depth, start_time, timeout):

        if (time.time() - start_time) > timeout:
            self.out_of_time = True
            return None

        # if depth > self.max_depth:
        #    self.max_depth = depth

        boardHash = state.getBoardHash()

        if boardHash in self.proven_win_states:
            return self.INFINITY

        if boardHash in self.proven_lost_states:
            return -self.INFINITY

        if boardHash not in self.moves_:
            cnnActive = state.isCNNMoveOrderingActive(previous_score)
            legalMoves = state.computePrunedMovesFromSubgames(cnnActive)
            if len(legalMoves) == 0:
                self.proven_lost_states.add(boardHash)
                return -self.INFINITY

            if depth != 0:
                # Play the first move anyway (because we need the winning move)
                result = self.handleProcessingSubgames(legalMoves, boardHash)
                if abs(result) == self.INFINITY:
                    return result

            legalMoves = self.cnnMoveOrdering(state, legalMoves, previous_score, cnnActive)
            self.moves_[boardHash] = legalMoves
        else:
            legalMoves = self.moves_[boardHash]

        isEndOfGame = len(legalMoves) == 0

        if isEndOfGame:
            self.proven_lost_states.add(boardHash)
            return -self.INFINITY

        self.nodes_visited.add(boardHash)

        for nextMove, score in legalMoves:
            savedHash = state.play(nextMove)
            nextStateHash = state.getBoardHash()

            if nextStateHash not in self.proven_lost_states:
                # Next State Win or unknown
                self.negamaxClobber1d(
                    state, score, depth + 1, start_time, timeout
                )

                if self.out_of_time:
                    return None
                state.undoMove(nextMove, savedHash)
            else:
                # LOSE for next player
                state.undoMove(nextMove, savedHash)
                self.proven_win_states.add(boardHash)
                self.proven_lost_states.remove(nextStateHash)
                if boardHash in self.moves_:
                    self.moves_.pop(boardHash)
                if depth == 0:
                    self.winningMove = nextMove
                return self.INFINITY

            if nextStateHash in self.proven_lost_states:
                self.proven_win_states.add(boardHash)
                self.proven_lost_states.remove(nextStateHash)
                if boardHash in self.moves_:
                    self.moves_.pop(boardHash)
                if depth == 0:
                    self.winningMove = nextMove
                return self.INFINITY

            """ END OF LOOP """

        self.proven_lost_states.add(boardHash)
        if boardHash in self.moves_:
            self.moves_.pop(boardHash)
        return -self.INFINITY

    def negamaxClobberGamePlay(self, state, start_time, timeout=300):
        self.out_of_time = False
        timeout = timeout * 0.98  # Safety barrier
        self.proven_lost_states = set()
        self.proven_win_states = set()
        self.moves_ = dict()
        self.nodes_visited = set()
        boardHash = state.getBoardHash()
        depth = 0
        outcome = self.negamaxClobber1d(
            state, 0, depth, start_time, timeout
        )
        if outcome == self.INFINITY:
            return self.PROVEN_WIN, self.winningMove, len(self.nodes_visited)

        elif outcome == -self.INFINITY and boardHash in self.proven_lost_states:
            return self.PROVEN_LOSS, None, len(self.nodes_visited)

        return self.UNKNOWN, None, len(self.nodes_visited)

    def getMaxDepth(self):
        return self.max_depth
