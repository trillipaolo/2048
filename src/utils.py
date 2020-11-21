import numpy as np
import inspect


def generate_rand_from_dict(generator, const_dict, num_rand):
    return generator.choice(list(const_dict.keys()), num_rand, p=list(const_dict.values()))


def get_bigger_number_from_probability_dictionary(dictionary):
    keys = np.array(list(dictionary.keys()))
    values = np.array(list(dictionary.values()))

    filtered_keys = keys[values > 0]

    return filtered_keys.max()


def check_probability_dictionary(dictionary, name):
    keys = np.array(list(dictionary.keys()))
    values = np.array(list(dictionary.values()))

    for idx, k in enumerate(keys):
        assert 0 <= values[idx] <= 1, f"{name}['{k}'] NOT IN [0, 1]"

    assert np.sum(values) == 1, f"{name} SUM != 1"
