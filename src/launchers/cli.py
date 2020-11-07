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


def main():
    last_key = np.zeros(4, dtype=int)
    reward = 0
    model = MDP_2048()
    model.initialize_state()
    print("SCORE:", reward)
    model.print_state()

    while not keyboard.is_pressed("esc"):

        action, last_key = get_action(last_key)

        if action is not None:
            model.transition_function(action)
            reward = model.reward_function()

            system('clear')
            print("SCORE:", reward)
            model.print_state()

    print("\nGAME ENDED, SCORE =", reward)


if __name__ == "__main__":
    main()
