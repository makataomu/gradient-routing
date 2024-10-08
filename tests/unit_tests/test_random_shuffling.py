import copy

import numpy as np


def test_random_shuffling():
    # Just to sanity check that the random number generator is working as expected
    # We use this in tinystories_era.py to shuffle the training data across runs.

    num_elements = 1000
    shared_seed = 10
    different_seed = 47

    data = list(range(num_elements))
    rng = np.random.default_rng(shared_seed)
    rng.shuffle(data)
    data_shuffled = copy.copy(data)

    data_2 = list(range(num_elements))
    rng_2 = np.random.default_rng(shared_seed)
    rng_2.shuffle(data_2)
    data_shuffled_2 = copy.copy(data_2)

    assert data_shuffled == data_shuffled_2

    data_3 = list(range(num_elements))
    rng_3 = np.random.default_rng(different_seed)
    rng_3.shuffle(data_3)
    data_shuffled_3 = copy.copy(data_3)
    assert data_shuffled != data_shuffled_3
