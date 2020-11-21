import gym
from gym import spaces
import numpy as np
from os import system
from datetime import datetime

from src import utils
from src.constants import constants as cts


class Env2048(gym.Env):
    metadata = {'render.modes': ['cli', 'gui']}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, seed=None):
        super(Env2048, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            0,
            np.prod(cts.BOARD_DIMENSION) + utils.get_bigger_number_from_probability_dictionary(cts.PROB_NUMBER_NEW_TILES),
            shape=cts.BOARD_DIMENSION,
            dtype=np.int64
        )

        self.__seed = seed
        self.__rand = np.random.RandomState(datetime.now().microsecond if seed is None else seed)

        self.__state = np.zeros(cts.BOARD_DIMENSION, dtype=int)
        self.__done = False

        self.__max_tile = 0
        self.__step_count = 0
        self.__reward = 0
        self.__score = 0

    def step(self, action):
        state = self.__state.copy()

        state = self.__rotate_before_compact(state, action)
        state, reward = self.__compact_tiles(state)
        state = self.__rotate_after_compact(state, action)

        not_changed = np.equal(self.__state, state).all()

        if not_changed:
            self.__reward = -1
        else:
            state = self.__add_new_tiles(state)
            done = self.__termination_state(state)

            self.__state = state
            self.__done = done

            self.__max_tile = state.max()
            self.__step_count += 1
            self.__reward = reward
            self.__score += reward

        return self.__state, self.__reward, self.__done, {}

    def reset(self):
        self.__rand = np.random.RandomState(datetime.now().microsecond if self.__seed is None else self.__seed)

        self.__state = np.zeros(cts.BOARD_DIMENSION, dtype=np.int64)
        self.__done = False

        self.__max_tile = 0
        self.__step_count = 0
        self.__reward = 0
        self.__score = 0

        self.__initialize_state()

        return self.__state

    def render(self, mode=cts.RENDER_CLI):
        if mode == cts.RENDER_CLI:
            self.__render_cli()
        elif mode == cts.RENDER_GUI:
            return
        else:
            super(Env2048, self).render(mode=mode)

    def __initialize_state(self):

        state = self.__state.copy()

        board_empty = np.where(state == 0)
        indices_empty = np.stack(board_empty).T

        num_new_cells = utils.generate_rand_from_dict(self.__rand, cts.PROB_NUMBER_INITIAL_TILES, 1)
        new_cells = self.__rand.choice(len(indices_empty), num_new_cells, replace=False)
        indices_new_cells = np.split(indices_empty[new_cells].T, 2)

        state[indices_new_cells] = utils.generate_rand_from_dict(self.__rand, cts.PROB_VALUE_INITIAL_TILES, num_new_cells)

        self.__state = state

        return

    def __add_new_tiles(self, state):
        board_empty = np.where(state == 0)
        num_empty = len(board_empty[0])
        if num_empty == 0:
            return state
        indices_empty = np.stack(board_empty).T

        num_new_cells = min(utils.generate_rand_from_dict(self.__rand, cts.PROB_NUMBER_NEW_TILES, 1), num_empty)
        new_cells = self.__rand.choice(len(indices_empty), num_new_cells, replace=False)
        indices_new_cells = np.split(indices_empty[new_cells].T, 2)

        state[indices_new_cells] = utils.generate_rand_from_dict(self.__rand, cts.PROB_VALUE_NEW_TILES, num_new_cells)

        return state

    @staticmethod
    def __termination_state(state):
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
        def compact_row(row, reward_in):
            length = len(row)
            row = row[np.where(row > 0)]
            for i in range(len(row) - 1):
                if row[i] == row[i + 1] and row[i] > 0:
                    row[i] += 1
                    row[i + 1] = 0
                    reward_in[0] += 2 << row[i] - 1
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

    def get_state(self):
        return self.__state.copy()

    def get_reward(self):
        return self.__reward

    def get_score(self):
        return self.__score

    def get_max_tile(self):
        return self.__max_tile

    def get_step_count(self):
        return self.__step_count

    def __render_cli(self):
        state = self.__state
        score = self.__score

        board = np.power(2, state)
        board[state == 0] = 0

        def print_line(length):
            print("+" + "-" * (length - 2) + "+")

        system('cls')
        print("SCORE:", score, '\n')

        str_board = str(board) \
            .replace("[[", "| ").replace("]]", " |") \
            .replace(" [", "| ").replace("]", " |") \
            .replace("0", " ")

        print_line(int(len(str_board) / cts.BOARD_DIMENSION[0]))
        print(str_board)
        print_line(int(len(str_board) / cts.BOARD_DIMENSION[0]))
