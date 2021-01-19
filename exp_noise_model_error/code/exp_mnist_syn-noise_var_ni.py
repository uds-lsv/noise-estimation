import numpy as np

from noise_estimation.datacode import load_mnist_data
from noise_estimation.syn_noise import pick_noise_function
from noise_estimation.empirical_var_ni import compute_avg_sample_error
from noise_estimation.theoretical_var_ni import expected_se
from noise_estimation.plotting import plot_empirical_vs_theoretical_graph


def experiment_sample_size(noise_type):
    random_state = np.random.RandomState(12345)
    num_times = 500
    noise_levels = [0.7, 0.4, 0.1]
    make_noisy_function = pick_noise_function(noise_type)
    start_n = 100
    stop_n = 1001
    step_n = 100
    ns = np.arange(start_n, stop_n, step_n)
    probabilities_yi = [1/10] * 10
    filename = f"mnist_var-ni_{noise_type}_over-n_{num_times}repetitions.pdf"

    clean_data, num_labels = load_mnist_data()

    emp_theo_triples = []
    legend_texts = []
    for i, noise_level in enumerate(noise_levels):
        noisy_data, true_noise_matrix = make_noisy_function(clean_data, noise_level, random_state)

        empirical_errors, empirical_error_bars = zip(*[compute_avg_sample_error(sample_size, clean_data, noisy_data,
                                                       true_noise_matrix, num_times, random_state)
                                                       for sample_size in ns])
        theoretical_errors = [expected_se(sample_size, true_noise_matrix, probabilities_yi) for sample_size in ns]

        emp_theo_triples.append((empirical_errors, empirical_error_bars, theoretical_errors))
        legend_texts.append(f"noise level {noise_level:.1f}")

    plot_empirical_vs_theoretical_graph(filename, "sample size $n$", "noise model error", ns,
                                        emp_theo_triples, show_legend=False, legend_texts=legend_texts)


def experiment_noise_level(noise_type):
    random_state = np.random.RandomState(12345)
    num_times = 500
    ns = [100, 500, 1000]
    make_noisy_function = pick_noise_function(noise_type)
    start_nl = 0
    stop_nl = 0.81
    step_nl = 0.1
    noise_levels = np.arange(start_nl, stop_nl, step_nl)
    probabilities_yi = [1 / 10] * 10
    filename = f"mnist_var-ni_{noise_type}_over-noise-level_{num_times}repetitions.pdf"

    clean_data, num_labels = load_mnist_data()

    emp_theo_triples = []
    legend_texts = []
    for i, n in enumerate(ns):
        empirical_errors = []
        empirical_error_bars = []
        theoretical_errors = []
        for noise_level in noise_levels:
            noisy_data, true_noise_matrix = make_noisy_function(clean_data, noise_level, random_state)
            empirical_error, empirical_error_bar = compute_avg_sample_error(n, clean_data, noisy_data,
                                                                            true_noise_matrix, num_times,
                                                                            random_state)
            empirical_errors.append(empirical_error)
            empirical_error_bars.append(empirical_error_bar)
            theoretical_errors.append(expected_se(n, true_noise_matrix, probabilities_yi))
        emp_theo_triples.append((empirical_errors, empirical_error_bars, theoretical_errors))
        legend_texts.append(f"sample size $n$ {n}")

    plot_empirical_vs_theoretical_graph(filename, "noise level", "noise model error", noise_levels,
                                        emp_theo_triples, show_legend=False, legend_texts=legend_texts)


if __name__ == "__main__":
    noise_types = ["single-flip", "uniform", "multi-flip"]
    for noise_type in noise_types:
        experiment_sample_size(noise_type)
        experiment_noise_level(noise_type)
