# Script of computing the expected squared error of the noise matrix
import numpy as np

def var_p(ni, p):
    return (p * (1 - p)) / ni


def expected_se_one_row(ni, ps):
    """ Computes the expected squared error for one row of the noise matrix

    Args:
        ni (int): The number of samples used for the estimation of this row of the noise matrix (n_i in the formula)
        ps (List[float]): The probabilities of the noise flips. This excludes (!)
                          the probability for the true label.
    """
    first_sum = np.sum([var_p(ni, p) for p in ps])
    second_sum = 0
    for i in range(len(ps)):
        for j in range(len(ps)):
            if i != j:
                second_sum += (-ps[i] * ps[j]) / ni
    return 2 * first_sum + second_sum


def expected_se_same_nis(ni, true_noise_matrix):
    """ Computes the expected squared error when all n_i are the same

    :param ni:
    :param noise_matrix:
    :return:
    """
    return expected_se_different_row_ns([ni] * len(true_noise_matrix), true_noise_matrix)

def expected_se_different_row_ns(nis, true_noise_matrix):
    """ Computes the expected squared error when the n_i are different for each row of the noise matrix

    :param nis:
    :param true_noise_matrix:
    :return:
    """
    total_sum = 0
    for row_index, row in enumerate(true_noise_matrix):
        ps = np.concatenate((row[:row_index], row[row_index + 1:]))  # excluding the probability for the true label
        total_sum += expected_se_one_row(nis[row_index], ps)
    return total_sum