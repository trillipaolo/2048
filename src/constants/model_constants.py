from src import utils

# BOARD INITIALIZATION PARAMETERS
BOARD_DIMENSION = (4, 4)
PROB_NUMBER_INITIAL_TILES = {
    1: 0,
    2: 1,
}
PROB_VALUE_INITIAL_TILES = {
    1: 0.9,     # 2^1 = 2
    2: 0.1,     # 2^2 = 4
}

# NEW TILES PARAMETERS
PROB_NUMBER_NEW_TILES = {
    1: 1,
    2: 0,
}
PROB_VALUE_NEW_TILES = {
    1: 0.9,     # 2^1 = 2
    2: 0.1,     # 2^2 = 4
}

# ACTIONS
MOVE_RIGHT = 0
MOVE_LEFT = 1
MOVE_UP = 2
MOVE_DOWN = 3

# RENDER MODES
RENDER_CLI = 'CLI'
RENDER_GUI = 'GUI'

# CHECKS
assert len(BOARD_DIMENSION) == 2, "BOARD_DIMENSION LENGTH != 2"
utils.check_probability_dictionary(PROB_NUMBER_INITIAL_TILES, "PROB_NUMBER_INITIAL_TILES")
utils.check_probability_dictionary(PROB_VALUE_INITIAL_TILES, "PROB_VALUE_INITIAL_TILES")
utils.check_probability_dictionary(PROB_NUMBER_NEW_TILES, "PROB_NUMBER_NEW_TILES")
utils.check_probability_dictionary(PROB_VALUE_NEW_TILES, "PROB_VALUE_NEW_TILES")
