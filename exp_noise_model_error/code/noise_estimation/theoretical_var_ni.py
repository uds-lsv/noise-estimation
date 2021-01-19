import numpy as np
import scipy.stats


def compute_prob_sum_yi(probability_yi, sample_size):
    # additional term obtained through the variance of sampling n_i/y_i
    summation = 0
    for r in range(1, sample_size):
        summation += scipy.stats.binom.pmf(r, sample_size, probability_yi) * 1 / r
    return summation


def var_p(p, prob_sum_yi):
    return (p * (1 - p)) * prob_sum_yi


def expected_se_one_row(ps, probability_yi, sample_size):
    """ Computes the expected Standard Error assuming a k-flip noise scenario.

    Args:
        ps (List[float]): The probabilities of the noise flips. This excludes (!)
                          the probability for the true label.
        probability_yi: The probability of the true labels y_i being sampled (assuming binomial distribution)
    """
    prob_sum_yi = compute_prob_sum_yi(probability_yi, sample_size)
    return np.sum([var_p(p, prob_sum_yi) for p in ps])

def expected_se(sample_size, noise_matrix, probability_yis):
    total_sum = 0
    for row_index, row in enumerate(noise_matrix):
        ps = row #np.concatenate((row[:row_index], row[row_index + 1:]))  # excluding the probability for the true label
        total_sum += expected_se_one_row(ps, probability_yis[row_index], sample_size)
    return total_sum
