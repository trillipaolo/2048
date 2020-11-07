import numpy as np

# BOARD INITIALIZATION PARAMETERS
BOARD_DIMENSION = (4, 4)
assert len(BOARD_DIMENSION) == 2, "BOARD_DIMENSION LENGTH != 2"
PROB_NUMBER_INITIAL_TILES = {
    1: 0,
    2: 1,
}
assert np.sum(list(PROB_NUMBER_INITIAL_TILES.values())) == 1, "PROB_NUMBER_INITIAL_TILES SUM != 1"
PROB_VALUE_INITIAL_TILES = {
    1: 0.9,     # 2^1 = 2
    2: 0.1,     # 2^2 = 4
}
assert np.sum(list(PROB_VALUE_INITIAL_TILES.values())) == 1, "PROB_VALUE_INITIAL_TILES SUM != 1"

# NEW TILES PARAMETERS
PROB_NUMBER_NEW_TILES = {
    1: 1,
    2: 0,
}
assert np.sum(list(PROB_NUMBER_NEW_TILES.values())) == 1, "PROB_NUMBER_NEW_TILES SUM != 1"
PROB_VALUE_NEW_TILES = {
    1: 0.9,     # 2^1 = 2
    2: 0.1,     # 2^2 = 4
}
assert np.sum(list(PROB_VALUE_NEW_TILES.values())) == 1, "PROB_VALUE_NEW_TILES SUM != 1"

# ACTIONS
MOVE_RIGHT = "RIGHT"
MOVE_LEFT = "LEFT"
MOVE_UP = "UP"
MOVE_DOWN = "DOWN"
