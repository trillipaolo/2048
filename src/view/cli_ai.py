from os import path
import sys
module_path = path.abspath(path.join('./../'))
if module_path not in sys.path:
    sys.path.append(module_path)


from src.controller import cli_ai
import pickle


if __name__ == "__main__":
    with open("../src/winners/ctrnn.pkl", 'rb') as f:
        winner_net = pickle.load(f)

    cli_ai.game(winner_net)
