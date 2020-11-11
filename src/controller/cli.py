import numpy as np
import keyboard
from os import system
from src.model.MDP_2048 import MDP_2048
from src.constants import constants as cts


def get_action(last):
    curr = np.zeros(4, dtype=int)
    if keyboard.is_pressed("right"):
        curr[0] = 1
    if keyboard.is_pressed("left"):
        curr[1] = 1
    if keyboard.is_pressed("up"):
        curr[2] = 1
    if keyboard.is_pressed("down"):
        curr[3] = 1

    if curr.sum() != 1 or np.equal(last, curr).all():
        return None, curr

    if curr[0] == 1:
        return cts.MOVE_RIGHT, curr
    if curr[1] == 1:
        return cts.MOVE_LEFT, curr
    if curr[2] == 1:
        return cts.MOVE_UP, curr
    if curr[3] == 1:
        return cts.MOVE_DOWN, curr


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


def game():
    last_key = np.zeros(4, dtype=int)

    model = MDP_2048()
    model.initialize_state()

    print_board(model.get_state(), model.get_score())

    while not keyboard.is_pressed("esc") and not model.termination_state():

        action, last_key = get_action(last_key)

        if action is not None:
            model.transition_function(action)

            print_board(model.get_state(), model.get_score())

    print("\nGAME ENDED, SCORE =", model.get_score())
