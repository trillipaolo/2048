import numpy as np
from datetime import datetime
from src import utils
from src.constants import constants as cts


class MDP_2048():

    def __init__(self, seed=None):

        self.rand = np.random.RandomState(datetime.now().microsecond if seed is None else seed)
        self.state = np.zeros(cts.BOARD_DIMENSION, dtype=int)
        self.actions = [cts.MOVE_RIGHT, cts.MOVE_LEFT, cts.MOVE_UP, cts.MOVE_DOWN]
        self.max_tile = 0
        self.step_count = 0
        self.score = 0
        self.reward = 0

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
        state, reward = self.__compact_tiles(state)
        state = self.__rotate_after_compact(state, action)

        if np.equal(self.state, state).all():
            return False

        state = self.__add_new_tiles(state)

        self.state = state
        self.reward = reward
        self.score += reward
        self.step_count += 1
        self.max_tile = state.max()

        return True

    def termination_state(self):
        state = self.state.copy()

        if (state == 0).any():
            return False
        if (np.diff(state, n=1, axis=0) == 0).any():
            return False
        if (np.diff(state, n=1, axis=1) == 0).any():
            return False
        return True

    @staticmethod
    def __rotate_before_compact(state, action):
        if action == cts.MOVE_RIGHT:
            return np.flip(state, axis=1)
        elif action == cts.MOVE_UP:
            return state.T
        elif action == cts.MOVE_DOWN:
            return np.flip(state.T, axis=1)
        else:
            return state

    @staticmethod
    def __compact_tiles(state):
        def compact_row(row, reward):
            length = len(row)
            row = row[np.where(row > 0)]
            for i in range(len(row) - 1):
                if row[i] == row[i + 1] and row[i] != 0:
                    row[i] += 1
                    row[i + 1] = 0
                    reward[0] += 2 ** row[i]
            row = row[np.where(row > 0)]
            row = np.pad(row, (0, length - len(row)), 'constant', constant_values=0)
            return row

        reward = [0]
        state_out = np.apply_along_axis(compact_row, 1, state, reward)

        return state_out, reward[0]

    @staticmethod
    def __rotate_after_compact(state, action):
        if action == cts.MOVE_RIGHT:
            return np.flip(state, axis=1)
        elif action == cts.MOVE_UP:
            return state.T
        elif action == cts.MOVE_DOWN:
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

    def get_state(self):
        return self.state.copy()

    def get_reward(self):
        return self.reward

    def get_score(self):
        return self.score

    def get_max_tile(self):
        return self.max_tile

    def get_step_count(self):
        return self.step_count

    def print_state(self):
        def print_line(length):
            print("+" + "-" * (length - 2) + "+")

        board = str(self.state)\
            .replace("[[", "| ").replace("]]", " |")\
            .replace(" [", "| ").replace("]", " |")\
            .replace("0", " ")

        print_line(int(len(board) / cts.BOARD_DIMENSION[0]))
        print(board)
        print_line(int(len(board) / cts.BOARD_DIMENSION[0]))
