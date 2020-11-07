import numpy as np
from src.constants import constants as cts


def generate_rand_from_dict(generator, const_dict, num_rand):
    return generator.choice(list(const_dict.keys()), num_rand, p=list(const_dict.values()))
