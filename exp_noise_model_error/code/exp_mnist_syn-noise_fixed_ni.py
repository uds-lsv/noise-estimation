import numpy as np

from noise_estimation.datacode import load_mnist_data
from noise_estimation.syn_noise import pick_noise_function
from noise_estimation.empirical_fixed_ni import compute_avg_sample_error
from noise_estimation.theoretical_fixed_ni import expected_se_same_nis
from noise_estimation.noisematrix import NoiseMatrix
from noise_estimation.plotting import plot_empirical_vs_theoretical_graph, plot_noise_matrix


def experiment_sample_size(noise_type):
    random_state = np.random.RandomState(12345)
    num_times = 500
    noise_levels = [0.7, 0.4, 0.1]
    make_noisy_function = pick_noise_function(noise_type)
    start_n = 10
    stop_n = 101
    step_n = 10
    nis = np.arange(start_n, stop_n, step_n)
    filename = f"mnist_{noise_type}_over-ni_{num_times}repetitions.pdf"

    clean_data, num_labels = load_mnist_data()

    emp_theo_triples = []
    legend_texts = []
    for i, noise_level in enumerate(noise_levels):
        noisy_data, true_noise_matrix = make_noisy_function(clean_data, noise_level, random_state)

        empirical_errors, empirical_error_bars = zip(*[compute_avg_sample_error(ni, clean_data, noisy_data,
                                                       true_noise_matrix, num_times, random_state) for ni in nis])
        theoretical_errors = [expected_se_same_nis(ni, true_noise_matrix) for ni in nis]

        emp_theo_triples.append((empirical_errors, empirical_error_bars, theoretical_errors))
        legend_texts.append(f"noise level {noise_level:.1f}")

    plot_empirical_vs_theoretical_graph(filename, "sample size $n_i$", "noise model error", nis,
                                        emp_theo_triples, show_legend=False, legend_texts=legend_texts)


def experiment_noise_level(noise_type):
    random_state = np.random.RandomState(12345)
    num_times = 500
    nis = [20, 50, 100]
    make_noisy_function = pick_noise_function(noise_type)
    start_nl = 0
    stop_nl = 0.81
    step_nl = 0.1
    noise_levels = np.arange(start_nl, stop_nl, step_nl)
    filename = f"mnist_{noise_type}_over-noise-level_{num_times}repetitions.pdf"

    clean_data, num_labels = load_mnist_data()

    emp_theo_triples = []
    legend_texts = []
    for i, ni in enumerate(nis):
        empirical_errors = []
        empirical_error_bars = []
        theoretical_errors = []
        for noise_level in noise_levels:
            noisy_data, true_noise_matrix = make_noisy_function(clean_data, noise_level, random_state)
            empirical_error, empirical_error_bar = compute_avg_sample_error(ni, clean_data, noisy_data,
                                                                            true_noise_matrix, num_times,
                                                                            random_state)
            empirical_errors.append(empirical_error)
            empirical_error_bars.append(empirical_error_bar)
            theoretical_errors.append(expected_se_same_nis(ni, true_noise_matrix))
        emp_theo_triples.append((empirical_errors, empirical_error_bars, theoretical_errors))
        legend_texts.append(f"sample size $n_i$ {ni}")

    plot_empirical_vs_theoretical_graph(filename, "noise level", "noise model error", noise_levels,
                                        emp_theo_triples, show_legend=False, legend_texts=legend_texts)


def visualize_example_noise_matrices():
    clean_data, num_labels = load_mnist_data()

    # uniform noise
    noisy_data, true_noise_matrix = pick_noise_function("uniform")(clean_data, 0.5, np.random.RandomState(12345))
    noise_matrix = NoiseMatrix.compute_noise_matrix(clean_data, noisy_data, num_labels)
    plot_noise_matrix("mnist_noise-matrix_uniform.pdf", noise_matrix, plot_bar=True)

    # single-flip noise
    noisy_data, true_noise_matrix = pick_noise_function("single-flip")(clean_data, 0.3, np.random.RandomState(12345))
    noise_matrix = NoiseMatrix.compute_noise_matrix(clean_data, noisy_data, num_labels)
    plot_noise_matrix("mnist_noise-matrix_single-flip.pdf", noise_matrix, plot_bar=True)

    # multi-flip noise
    noisy_data, true_noise_matrix = pick_noise_function("multi-flip")(clean_data, 0.4, np.random.RandomState(12345))
    noise_matrix = NoiseMatrix.compute_noise_matrix(clean_data, noisy_data, num_labels)
    plot_noise_matrix("mnist_noise-matrix_multi-flip.pdf", noise_matrix, plot_bar=True)


if __name__ == "__main__":
    noise_types = ["single-flip", "uniform", "multi-flip"]
    for noise_type in noise_types:
        experiment_sample_size(noise_type)
        #experiment_noise_level(noise_type)
    #visualize_example_noise_matrices()
