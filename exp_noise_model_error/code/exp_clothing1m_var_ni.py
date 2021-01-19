import numpy as np

from noise_estimation.datacode import load_clothing1m_data
from noise_estimation.empirical_var_ni import compute_avg_sample_error, estimate_probability_yis
from noise_estimation.theoretical_var_ni import expected_se
from noise_estimation.noisematrix import NoiseMatrix
from noise_estimation.plotting import plot_empirical_vs_theoretical_graph, plot_noise_matrix


def experiment_sample_size():
    random_state = np.random.RandomState(12345)
    num_times = 500
    start_n = 100
    stop_n = 1001
    step_n = 100
    ns = np.arange(start_n, stop_n, step_n)
    filename = f"clothing1m_var-ni_over-n_{num_times}repetitions.pdf"

    clean_data, noisy_data, label_idx_to_label_name_map, num_labels = load_clothing1m_data()
    true_noise_matrix = NoiseMatrix.compute_noise_matrix(clean_data, noisy_data, num_labels)

    empirical_errors, empirical_error_bars = zip(*[compute_avg_sample_error(n, clean_data, noisy_data,
                                                                            true_noise_matrix, num_times,
                                                                            random_state) for n in ns])

    probability_yis = estimate_probability_yis(clean_data, len(true_noise_matrix))
    theoretical_errors = [expected_se(n, true_noise_matrix, probability_yis) for n in ns]

    plot_empirical_vs_theoretical_graph(filename, "sample size $n$", "noise model error", ns,
                                        [(empirical_errors, empirical_error_bars, theoretical_errors)],
                                        show_legend=False)


if __name__ == "__main__":
    experiment_sample_size()
