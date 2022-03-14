# Represent infinity by a value that is
# larger than all game values that can occur,
# but small enough to be treated as a Python 3 int
INFINITY = 10000

# Encoding of game results
PROVEN_WIN = 10000
PROVEN_LOSS = -10000
UNKNOWN = -5  # heuristic score, set to 1 to see a difference from draw
DRAW = 0
