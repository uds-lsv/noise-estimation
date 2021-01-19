import numpy as np

from noise_estimation.datacode import load_clothing1m_data
from noise_estimation.empirical_fixed_ni import compute_avg_sample_error
from noise_estimation.theoretical_fixed_ni import expected_se_same_nis
from noise_estimation.noisematrix import NoiseMatrix
from noise_estimation.plotting import plot_empirical_vs_theoretical_graph, plot_noise_matrix


def experiment_sample_size():
    random_state = np.random.RandomState(12345)
    num_times = 500
    start_n = 10
    stop_n = 101
    step_n = 10
    nis = np.arange(start_n, stop_n, step_n)
    filename = f"clothing1m_over-ni_{num_times}repetitions.pdf"

    clean_data, noisy_data, label_idx_to_label_name_map, num_labels = load_clothing1m_data()
    true_noise_matrix = NoiseMatrix.compute_noise_matrix(clean_data, noisy_data, num_labels)

    empirical_errors, empirical_error_bars = zip(*[compute_avg_sample_error(ni, clean_data, noisy_data,
                                                                            true_noise_matrix, num_times,
                                                                            random_state) for ni in nis])
    theoretical_errors = [expected_se_same_nis(ni, true_noise_matrix) for ni in nis]

    plot_empirical_vs_theoretical_graph(filename, "sample size $n_i$", "noise model error", nis,
                                        [(empirical_errors, empirical_error_bars, theoretical_errors)],
                                        show_legend=False)


def visualize_noise_matrices():
    clean_data, noisy_data, label_idx_to_label_name_map, num_labels = load_clothing1m_data()
    true_noise_matrix = NoiseMatrix.compute_noise_matrix(clean_data, noisy_data, num_labels)
    plot_noise_matrix("clothing1m_noise-matrix.pdf", true_noise_matrix, plot_bar=True,
                      idx_to_label_name_map=label_idx_to_label_name_map)


if __name__ == "__main__":
    experiment_sample_size()
    visualize_noise_matrices()