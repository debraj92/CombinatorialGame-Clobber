import time


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

    def cnnMoveOrdering(self, legalMoves):
        moves = []
        for move_set, _, _, _, _, in legalMoves:
            for nextMove in move_set:
                moves.append(nextMove)

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

    def negamaxClobber1d(self, state, alpha, beta, depth, start_time, timeout):

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
            legalMoves = state.computePrunedMovesFromSubgames()
            if len(legalMoves) == 0:
                self.proven_lost_states.add(boardHash)
                return -self.INFINITY

            if depth != 0:
                # Play the first move anyway (because we need the winning move)
                result = self.handleProcessingSubgames(legalMoves, boardHash)
                if abs(result) == self.INFINITY:
                    return result

            self.moves_[boardHash] = legalMoves
        else:
            legalMoves = self.moves_[boardHash]

        isEndOfGame = len(legalMoves) == 0

        if isEndOfGame:
            self.proven_lost_states.add(boardHash)
            return -self.INFINITY

        self.nodes_visited.add(boardHash)

        for move_set, _, _, _, _, in legalMoves:
            for nextMove in move_set:
                savedHash = state.play(nextMove)
                nextStateHash = state.getBoardHash()

                if nextStateHash not in self.proven_lost_states:
                    # Next State Win or unknown
                    self.negamaxClobber1d(
                        state, -beta, -alpha, depth + 1, start_time, timeout
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
        depth = 1000
        outcome = self.negamaxClobber1d(
            state, -self.INFINITY, self.INFINITY, 0, start_time, timeout
        )
        if outcome == self.INFINITY:
            return self.PROVEN_WIN, self.winningMove, len(self.nodes_visited)

        elif outcome == -self.INFINITY and boardHash in self.proven_lost_states:
            return self.PROVEN_LOSS, None, len(self.nodes_visited)

        return self.UNKNOWN, None, len(self.nodes_visited)

    def getMaxDepth(self):
        return self.max_depth
