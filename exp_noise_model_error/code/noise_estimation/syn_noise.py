import copy
import numpy as np


def make_noisy_general(clean_data, noise_matrix, random_state):
    """ Perturbs the MNIST labels based on the probabilities of the given noise matrix

    Args:
        clean_data: list of instances
        noise_matrix: defines the noise process
        random_state: for reproducibility

    Returns:
        A perturbed copy of clean_data (the noisy_data)
    """
    for row in noise_matrix:
        assert np.isclose(np.sum(row), 1)

    noisy_data = copy.deepcopy(clean_data)
    for instance in noisy_data:
        probability_row = noise_matrix[instance.label]
        instance.label = random_state.choice(10, p=probability_row)
    return noisy_data


def make_noisy_single_flip(clean_data, noise_level, random_state):
    """ Perturbs the MNIST labels using a single-flip noise

        Noise permutation used by Reed et al. (https://arxiv.org/pdf/1412.6596.pdf)
        (they flip axis for true and noisy in their figure 2)

        Was also used by Goldberger & Ben-Reuven (https://openreview.net/forum?id=H12GRgcxg and code
        https://github.com/udibr/noisy_labels/blob/master/mnist-simple.ipynb)

        Args:
            clean_data: list of instances
            noise_level: the probability with which the true label is flipped
            random_state: for reproducibility

        Returns:
            A perturbed copy of clean_data (the noisy_data) and the true noise matrix
    """
    flips = np.array([7, 9, 0, 4, 2, 1, 3, 5, 6, 8])

    true_noise_matrix = np.zeros((10, 10))
    for true_label in range(10):
        true_noise_matrix[true_label][true_label] = 1 - noise_level
        true_noise_matrix[true_label][flips[true_label]] = noise_level

    noisy_data = make_noisy_general(clean_data, true_noise_matrix, random_state)

    return noisy_data, true_noise_matrix


def make_noisy_uniform(clean_data, noise_level, random_state):
    """ Perturbs the MNIST labels using uniform noise.

    Args:
        clean_data: list of instances
        noise_level: the overall noise level.
                     k*epsilon, where epsilon is the probability
                     that one specific non-true label is picked
                     (with k non-true labels)
        random_state: for reproducibility

    Returns:
        A perturbed copy of clean_data (the noisy_data) and the true noise matrix
    """
    clean_label_probability = 1 - noise_level
    uniform_noise_probability = noise_level / 9  # distribute noise_level across all other labels

    true_noise_matrix = np.empty((10, 10))
    true_noise_matrix.fill(uniform_noise_probability)
    for true_label in range(10):
        true_noise_matrix[true_label][true_label] = clean_label_probability

    noisy_data = make_noisy_general(clean_data, true_noise_matrix, random_state)

    return noisy_data, true_noise_matrix


def make_noisy_multi_flip(clean_data, noise_level, random_state):
    """ Perturbs the MNIST labels using a (specific) multi-flip noise.

    The noise is inspired by possible, actual confusions on MNIST
    (e.g. confusing an 8 and a 9).

    Args:
        clean_data: list of instances
        noise_level: the overall noise level. Each row distributes the
        noise differently.
        random_state: for reproducibility

    Returns:
        A perturbed copy of clean_data (the noisy_data) and the true noise matrix
    """
    e = noise_level
    true_noise_matrix = np.array([
        [1 - e, 0, 0, 0, 0, 0, 0, 0, e / 2, e / 2],
        [0, 1 - e, 0, 0, 0, 0, 0, e, 0, 0],
        [e / 3, 0, 1 - e, 2 * e / 3, 0, 0, 0, 0, 0, 0],
        [0, 0, e / 2, 1 - e, 0, 0, 0, 0, e / 2, 0],
        [e / 5, e / 5, 0, 0, 1 - e, e / 5, e / 5, 0, e / 5, 0],
        [0, 0, 0, 0, 0, 1 - e, e / 2, 0, e / 2, 0],
        [0, 0, 0, 0, 0, e / 2, 1 - e, 0, e / 2, 0],
        [0, 2 * e / 6, 0, 0, e / 6, 0, 0, 1 - e, 0, 3 * e / 6],
        [0, 0, 3 * e / 4, 0, 0, 0, 0, 0, 1 - e, e / 4],
        [e / 3, 0, 0, 0, e / 3, 0, 0, 0, e / 3, 1 - e]
    ])

    noisy_data = make_noisy_general(clean_data, true_noise_matrix, random_state)

    return noisy_data, true_noise_matrix


def pick_noise_function(noise_type):
    if noise_type == "single-flip":
        return make_noisy_single_flip
    elif noise_type == "uniform":
        return make_noisy_uniform
    elif noise_type == "multi-flip":
        return make_noisy_multi_flip
    else:
        raise Exception(f"Noise type {noise_type} unknown.")