import numpy as np

from noise_estimation.datacode import load_ner_data_specific_round
from noise_estimation.empirical_fixed_ni import compute_avg_sample_error
from noise_estimation.theoretical_fixed_ni import expected_se_same_nis
from noise_estimation.noisematrix import NoiseMatrix
from noise_estimation.plotting import plot_empirical_vs_theoretical_graph, plot_noise_matrix


def experiment_sample_size(labeling_rounds):
    random_state = np.random.RandomState(12345)
    num_times = 500
    start_n = 10
    stop_n = 101
    step_n = 10
    nis = np.arange(start_n, stop_n, step_n)
    round_identifier = ','.join([str(labeling_round) for labeling_round in labeling_rounds])
    filename = f"ner_rounds{round_identifier}_over-ni_{num_times}repetitions.pdf"

    emp_theo_triples = []
    legend_texts = []
    for labeling_round in labeling_rounds:
        clean_data, noisy_data, label_idx_to_label_name_map, label_name_to_idx_map, \
            num_labels = load_ner_data_specific_round(labeling_round)
        true_noise_matrix = NoiseMatrix.compute_noise_matrix(clean_data, noisy_data, num_labels)

        empirical_errors, empirical_error_bars = zip(*[compute_avg_sample_error(ni, clean_data, noisy_data,
                                                                                true_noise_matrix, num_times,
                                                                                random_state) for ni in nis])
        theoretical_errors = [expected_se_same_nis(ni, true_noise_matrix) for ni in nis]

        emp_theo_triples.append((empirical_errors, empirical_error_bars, theoretical_errors))
        legend_texts.append(f"label set {labeling_round}")
    plot_empirical_vs_theoretical_graph(filename, "sample size $n_i$", "noise model error", nis,
                                        emp_theo_triples, show_legend=False, legend_texts=legend_texts)


def visualize_noise_matrices(labeling_round):
    clean_data, noisy_data, label_idx_to_label_name_map, label_name_to_idx_map, \
        num_labels = load_ner_data_specific_round(labeling_round)
    true_noise_matrix = NoiseMatrix.compute_noise_matrix(clean_data, noisy_data, num_labels)
    plot_noise_matrix(f"ner_round{labeling_round}_noise-matrix.pdf", true_noise_matrix, plot_bar=True,
                      idx_to_label_name_map=label_idx_to_label_name_map)


if __name__ == "__main__":
    current_labeling_rounds = [1, 4, 7]
    experiment_sample_size(current_labeling_rounds)

    #for label in [2, 3, 5, 6]:
    #    experiment_sample_size([label])

    #for current_labeling_round in range(1, 8):
    #    visualize_noise_matrices(current_labeling_round)
