import numpy as np
import keyboard
from os import system
from src.model.MDP_2048 import MDP_2048
from src.constants import constants as cts
import neat
import neat.math_util
import time


def print_board(state, score):
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


def get_action(probabilities):
    prob = np.array(probabilities)
    index = prob.argsort()

    action_dict = {
        0: cts.MOVE_RIGHT,
        1: cts.MOVE_LEFT,
        2: cts.MOVE_UP,
        3: cts.MOVE_DOWN
    }

    actions = [action_dict[x] for x in index]

    return actions


def game(net, seed=None):

    model = MDP_2048(seed)
    model.initialize_state()

    print_board(model.get_state(), model.get_score())

    while not keyboard.is_pressed("esc") and not model.termination_state():

        output = net.activate(model.get_state().flatten())
        actions = get_action(neat.math_util.softmax(output))

        action_index = 0
        while not model.transition_function(actions[action_index]):
            action_index += 1

        print_board(model.get_state(), model.get_score())

        # time.sleep(0.1)

    print("\nGAME ENDED, SCORE =", model.get_score())

    while keyboard.is_pressed("esc"):
        pass
