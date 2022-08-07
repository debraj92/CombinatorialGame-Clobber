import time
import keras.models
import numpy as np
import tensorflow as tf
from rl import deployable_rl_agent

class PlayClobber:
    # Transposition table to avoid re-computation of proven lost games
    proven_lost_states = set()

    # Transposition table to avoid re-computation of proven won games
    proven_win_states = set()

    moves_ = {}

    nodes_visited = set()

    out_of_time = False

    max_depth = 0

    def __init__(self, move_ordering):
        self.PROVEN_WIN = 10000
        self.PROVEN_LOSS = -10000
        self.UNKNOWN = -5
        self.PROVEN = 0
        self.DISPROVEN = 1
        self.INFINITY = 10000
        self.winningMove = ()
        self.move_ordering = move_ordering
        if self.move_ordering["cnn"]:
            self.model_black_interpreter = self.modelInferenceInit("./cnn-models/best3/clobber-black-cnn.tflite")
            self.model_white_interpreter = self.modelInferenceInit("./cnn-models/best3/clobber-white-cnn.tflite")
        elif self.move_ordering["rl"]:
            self.rl_model = deployable_rl_agent.DeployableAgent("./rl_models/model_size_40")

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
        cnnOrdering = cnnOrdering and state.count_legal_moves > 17
        for move_set, _, win, lose, _ in legalMoves:
            for nextMove in move_set:
                if cnnOrdering:
                    prediction = self.evaluateMove(state, nextMove)
                    if prediction < -0.7:
                        cnnOrdering = False
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

    def noMoveOrdering(self, legalMoves):
        moves = []
        for move_set, _, win, lose, _ in legalMoves:
            for nextMove in move_set:
                moves.append((nextMove, 0))
        return moves

    def defaultMoveOrdering(self, legalMoves, previous_score):
        moves = []
        for move_set, _, win, lose, _ in legalMoves:
            for nextMove in move_set:
                if win and not lose:
                    moves.append((nextMove, 0.6))
                elif not win and lose:
                    moves.append((nextMove, -0.6))
                else:
                    moves.append((nextMove, 1/len(move_set)))
        
        moves = sorted(moves, key=lambda x: x[1])
        return moves

    def rlMoveOrdering(self, state, legalMoves, previous_score):
        # TODO: Simplify bigger boards to less than model.board_size?
        # RL is active only if the model can accomodate the board
        # AND the number of tokens on the board is at least 20.
        if (len(state.board) <= self.rl_model.board_size) and (len(state.player_positions) + len(state.opponent_positions) >= 20):
            board = state.getPaddedBoard(self.rl_model.board_size)
            player = state.toPlay
            legal_moves = [move for legal_moves in legalMoves for move in legal_moves[0]]
            # Returns an ordered list of moves of form (src,target)
            moves = self.rl_model.move_ordering(board, player, legal_moves)
            return [(move, 1) for move in moves]
        else:
            return self.defaultMoveOrdering(legalMoves, previous_score)

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
            cnnActive = state.isCNNMoveOrderingActive(previous_score) if self.move_ordering["cnn"] else False
            legalMoves = state.computePrunedMovesFromSubgames(cnnActive)
            if len(legalMoves) == 0:
                self.proven_lost_states.add(boardHash)
                return -self.INFINITY

            if depth != 0:
                # Play the first move anyway (because we need the winning move)
                result = self.handleProcessingSubgames(legalMoves, boardHash)
                if abs(result) == self.INFINITY:
                    return result

            if self.move_ordering["cnn"] or self.move_ordering["default"]:
                legalMoves = self.cnnMoveOrdering(state, legalMoves, previous_score, cnnActive)
            elif self.move_ordering["rl"]:
                legalMoves = self.rlMoveOrdering(state, legalMoves, previous_score)
            elif self.move_ordering["none"]:
                legalMoves = self.noMoveOrdering(legalMoves)

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
