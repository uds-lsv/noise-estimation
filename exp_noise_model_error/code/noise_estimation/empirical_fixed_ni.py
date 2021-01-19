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
        sample_size: n_i in the paper, number of samples for each clean class
        clean_data: the list of all clean instances
        noisy_data: the list of all noisy instances (must be parallel to the clean instances list)
        true_noise_matrix: the true noise matrix (to compute the difference to the estimated one)
        random_state: random_state for reproducibility
    """
    assert len(clean_data) == len(noisy_data)

    # for each label, sample num_labels instances
    num_labels = len(true_noise_matrix)
    sampled_indices = []
    for row in range(num_labels):
        sampled_indices.extend(sample_indices_one_row(sample_size, clean_data, row, random_state))

    assert len(set(sampled_indices)) == len(sampled_indices)

    sample_clean_data = np.array(clean_data)[sampled_indices]
    sample_noisy_data = np.array(noisy_data)[sampled_indices]

    # estimate noise matrix
    sample_noise_matrix = NoiseMatrix.compute_noise_matrix(sample_clean_data, sample_noisy_data, num_labels)

    # compute error
    error = se(true_noise_matrix, sample_noise_matrix)
    return error


def sample_indices_one_row(sample_size, clean_data, row, random_state):
    """ Sampling indices for a specific clean label

    Args:
        sample_size: n_i in the paper, number of samples for the specified clean label value
        clean_data: the list of clean instances
        row: the row in the noise matrix, i.e. the clean label value
        random_state: random_state for reproducibility
    """
    row_indices = []
    for index, instance in enumerate(clean_data):
        if instance.label == row:
            row_indices.append(index)

    assert len(row_indices) >= sample_size

    return random_state.choice(row_indices, sample_size, replace=False)