import numpy as np
from src.model.MDP_2048 import MDP_2048
from src.constants import constants as cts
from datetime import datetime
import neat
import neat.math_util
import pickle
from joblib import Parallel, delayed


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = game(net)


def eval_genomes_parallel(genomes, config):

    def eval_genome(genome, config, seeds):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        rewards = Parallel(n_jobs=len(seeds))(
            delayed(game)(net, s, False) for s in seeds
        )

        return float(np.mean(rewards))

    rand_state = np.random.RandomState(datetime.now().microsecond)
    seeds = rand_state.randint(np.iinfo(np.int32).max, size=10)

    fitnesses = Parallel(n_jobs=32)(
        delayed(eval_genome)(genome, config, seeds) for genome_id, genome in genomes
    )

    for index, (genome_id, genome) in enumerate(genomes):
        genome.fitness = fitnesses[index]


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


def game(net, seed=None, return_max=False):

    model = MDP_2048(seed)
    model.initialize_state()

    while not model.termination_state():

        output = net.activate(model.get_state().flatten())
        actions = get_action(neat.math_util.softmax(output))

        action_index = 0
        while not model.transition_function(actions[action_index]):
            action_index += 1

    if return_max:
        return int(model.get_score()), model.get_max_tile()
    else:
        return int(model.get_score())


def run(config_file, winner_path):
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
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes_parallel, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    with open(winner_path, 'wb') as f:
        pickle.dump(winner_net, f)


def take_last(config_file, check_name):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Checkpointer.restore_checkpoint(check_name)
    winner = p.run(eval_genomes, 1)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    with open("../winners/ctrnn-parallel.pkl", 'wb') as f:
        pickle.dump(winner_net, f)


def test_result(model_path):
    with open(model_path, 'rb') as f:
        winner_net = pickle.load(f)

    for i in range(10):
        score, max_tile = game(winner_net, return_max=True)
        print(f"In game {i + 1}/10 winner_net has scored: {score}, with max of {2 ** max_tile}")


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-feedforward')

    run("../config/config-ctrnn", "../winners/ctrnn_parallel.pkl")
    test_result("../winners/ctrnn-parallel.pkl")
    # take_last("../../config/config-ctrnn", "neat-checkpoint-139")
