import numpy as np
from datetime import datetime
from src import utils
from src.constants import constants as cts


class MDP_2048():

    def __init__(self, seed=None):

        self.rand = np.random.RandomState(datetime.now().microsecond if seed is None else seed)
        self.state = np.zeros(cts.BOARD_DIMENSION, dtype=int)

        self.actions = [cts.MOVE_RIGHT, cts.MOVE_LEFT, cts.MOVE_UP, cts.MOVE_DOWN]

    def initialize_state(self):

        state = self.state.copy()

        board_empty = np.where(state == 0)
        indices_empty = np.stack(board_empty).T

        num_new_cells = utils.generate_rand_from_dict(self.rand, cts.PROB_NUMBER_INITIAL_TILES, 1)
        new_cells = self.rand.choice(len(indices_empty), num_new_cells, replace=False)
        indices_new_cells = np.split(indices_empty[new_cells].T, 2)

        state[indices_new_cells] = utils.generate_rand_from_dict(self.rand, cts.PROB_VALUE_INITIAL_TILES, num_new_cells)

        self.state = state

        return

    def transition_function(self, action):
        state = self.state.copy()

        state = self.__rotate_before_compact(state, action)
        state = self.__compact_tiles(state)
        state = self.__rotate_after_compact(state, action)

        if np.equal(self.state, state).all():
            return

        state = self.__add_new_tiles(state)

        self.state = state

        return

    def reward_function(self):
        return self.state.sum()

    @staticmethod
    def __rotate_before_compact(state, action):
        if action == cts.MOVE_RIGHT:
            return np.flip(state, axis=1)
        elif action == cts.MOVE_UP:
            return state.T
        elif action == cts.MOVE_UP:
            return np.flip(state.T, axis=1)
        else:
            return state

    @staticmethod
    def __compact_tiles(state):
        def compact_row(row):
            length = len(row)
            for i in range(length - 1):
                if row[i] == row[i + 1] & row[i] != 0:
                    row[i] += 1
                    row[i + 1] = 0
            row = row[np.where(row > 0)]
            row = np.pad(row, (0, length - len(row)), 'constant', constant_values=0)
            return row

        return np.apply_along_axis(compact_row, 1, state)

    @staticmethod
    def __rotate_after_compact(state, action):
        if action == cts.MOVE_RIGHT:
            return np.flip(state, axis=1)
        elif action == cts.MOVE_UP:
            return state.T
        elif action == cts.MOVE_UP:
            return np.flip(state, axis=1).T
        else:
            return state

    def __add_new_tiles(self, state):

        board_empty = np.where(state == 0)
        num_empty = len(board_empty[0])
        if num_empty == 0:
            return state
        indices_empty = np.stack(board_empty).T

        num_new_cells = min(utils.generate_rand_from_dict(self.rand, cts.PROB_NUMBER_NEW_TILES, 1), num_empty)
        new_cells = self.rand.choice(len(indices_empty), num_new_cells, replace=False)
        indices_new_cells = np.split(indices_empty[new_cells].T, 2)

        state[indices_new_cells] = utils.generate_rand_from_dict(self.rand, cts.PROB_VALUE_NEW_TILES, num_new_cells)

        return state

    def print_state(self):
        print(" " + str(self.state).replace("[", "").replace("]", ""))
