from .empirical import *
from .noisematrix import NoiseMatrix


def compute_avg_sample_error(sample_size, clean_data, noisy_data, true_noise_matrix, num_times,
                                     random_state):
    """ Repeats compute_sample_error() num_times times and returns the mean.
    """
    return compute_avg_sample_error_general(compute_sample_error, sample_size, clean_data, noisy_data,
                                            true_noise_matrix, num_times, random_state)


def compute_sample_error(sample_size, clean_data, noisy_data, true_noise_matrix, random_state):
    """ Computes the empirical sample error.

    Args:
        sample_size: n, overall sample size
        clean_data: the list of clean instances
        noisy_data: the list of noisy instances (must be parallel to the clean instances list)
        true_noise_matrix: the true noise matrix (to compute the difference to the estimated one)
        random_state: random_state for reproducibility
    """
    sampled_indices = sample_indices(sample_size, clean_data, random_state)

    sample_clean_data = np.array(clean_data)[sampled_indices]
    sample_noisy_data = np.array(noisy_data)[sampled_indices]

    num_labels = len(true_noise_matrix)

    # estimate noise matrix
    sample_noise_matrix = NoiseMatrix.compute_noise_matrix(sample_clean_data, sample_noisy_data, num_labels)

    # compute error
    error = se(true_noise_matrix, sample_noise_matrix)
    return error


def sample_indices(sample_size, clean_data, random_state):
    sampled_indices = random_state.choice(len(clean_data), sample_size, replace=False)
    assert len(set(sampled_indices)) == len(sampled_indices)
    return sampled_indices


def estimate_probability_yis(clean_data, num_labels):
    """ Estimate the probability of a clean label y=i being sampled by empirically counting the clean labels
    """
    prob_yis = np.zeros(num_labels)
    for instance in clean_data:
        prob_yis[instance.label] += 1
    prob_yis /= np.sum(prob_yis)
    return prob_yis
