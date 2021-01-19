import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm


def plot_empirical_vs_theoretical_graph(filename, x_label, y_label, x_values, emp_theo_triples, show_legend=True,
                                        legend_texts=None):
    font = {'family': 'serif',
            'size': 16}
    plt.rc('font', **font)

    if legend_texts is None:
        emp_legend_texts = ["empirical" for _ in range(len(emp_theo_triples))]
        theo_legend_texts = ["theoretical" for _ in range(len(emp_theo_triples))]
    else:
        emp_legend_texts = [f"empirical ({legend_text})" for legend_text in legend_texts]
        theo_legend_texts = [f"theoretical ({legend_text})" for legend_text in legend_texts]

    cmap = matplotlib.cm.get_cmap("Paired").colors
    colors_theo = [cmap[0], cmap[2], cmap[6]]
    colors_emp = [cmap[1], cmap[3], cmap[7]]

    fig, ax = plt.subplots()

    for i, (empirical_values, empirical_error_bar_values, theoretical_values) in enumerate(emp_theo_triples):
        plt.errorbar(x_values, empirical_values, empirical_error_bar_values, fmt=" ", color="gray", alpha="0.5")
        ax.plot(x_values, theoretical_values, "o", label=theo_legend_texts[i], color=colors_theo[i], markersize=7)
        ax.plot(x_values, empirical_values, "+", label=emp_legend_texts[i], color=colors_emp[i], markersize=10)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join("..", "plots", filename), bbox_inches='tight')
    plt.close()


def plot_noise_matrix(filename, matrix, plot_bar=True, idx_to_label_name_map=None):
    font = {'family': 'serif',
            'size': 16}
    plt.rc('font', **font)

    plt.matshow(matrix, vmin=0, vmax=1, interpolation="none", cmap=plt.cm.Blues)
    plt.xlabel("noisy label $\\hat{y}$")
    plt.ylabel("clean label $y$")
    if plot_bar:
        plt.colorbar(fraction=0.046, pad=0.04)

    if idx_to_label_name_map is not None:
        tick_marks = np.arange(len(idx_to_label_name_map))
        label_names = [idx_to_label_name_map[idx] for idx in tick_marks]
        plt.xticks(tick_marks, label_names, rotation=90)
        plt.yticks(tick_marks, label_names)

    plt.savefig(os.path.join("..", "plots", filename), bbox_inches="tight")
    plt.close()
