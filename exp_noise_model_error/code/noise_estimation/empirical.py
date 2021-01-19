import numpy as np


def se(mat1, mat2):
    """ Squared error """
    return np.sum((mat1 - mat2) ** 2)


def standard_error_of_mean(samples):
    """ Computes the standard error of the mean.

    Computes s / sqrt(n) where s is the sample standard deviation and n is the number of samples.
    """
    return np.std(samples) / np.sqrt(len(samples))

def standard_deviation(samples):
    return np.std(samples)


def compute_avg_sample_error_general(error_function, sample_size, clean_data, noisy_data, true_noise_matrix, num_times,
                                     random_state):
    """ Repeats compute_sample_error() num_times times and returns the mean.    
    """
    errors = []
    for i in range(num_times):
        errors.append(error_function(sample_size, clean_data, noisy_data, true_noise_matrix, random_state))
    return np.mean(errors), standard_deviation(errors)
