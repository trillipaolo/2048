import numpy as np
from src.model.MDP_2048 import MDP_2048
from src.constants import constants as cts
import neat
import neat.math_util
import pickle


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness, _ = game(net, 33)


def get_action(probabilities):
    prob = np.array(probabilities)
    index = prob.argmax()

    if index == 0:
        return cts.MOVE_RIGHT
    if index == 1:
        return cts.MOVE_LEFT
    if index == 2:
        return cts.MOVE_UP
    if index == 3:
        return cts.MOVE_DOWN


def game(net, seed=None):
    reward = 0
    model = MDP_2048(seed)
    model.initialize_state()
    last_state = model.state.copy()
    n_max_equal_state = 5
    n_equal_last_state = 0

    while n_equal_last_state < n_max_equal_state:

        output = net.activate(last_state.flatten())

        action = get_action(neat.math_util.softmax(output))

        model.transition_function(action)
        reward = model.get_reward()

        curr_state = model.state

        if np.equal(curr_state, last_state).all():
            n_equal_last_state += 1
        else:
            n_equal_last_state = 0
        last_state = curr_state.copy()

    return reward, last_state.max()


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    with open("../winners/ctrnn.pkl", 'wb') as f:
        pickle.dump(winner_net, f)

    for i in range(10):
        score, max_tile = game(winner_net)
        print(f"In game {i+1}/10 winner_net has scored: {score}, with max of {2 ** max_tile}")


def test_result(model_path):
    with open(model_path, 'rb') as f:
        winner_net = pickle.load(f)

    for i in range(10):
        score, max_tile = game(winner_net)
        print(f"In game {i + 1}/10 winner_net has scored: {score}, with max of {2 ** max_tile}")


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-feedforward')
    run("../../config/config-ctrnn")
    test_result("../winners/ctrnn.pkl")
