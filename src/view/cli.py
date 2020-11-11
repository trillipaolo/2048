from os import path
import sys
module_path = path.abspath(path.join('./../'))
if module_path not in sys.path:
    sys.path.append(module_path)


from src.controller import cli


if __name__ == "__main__":
    cli.game()
