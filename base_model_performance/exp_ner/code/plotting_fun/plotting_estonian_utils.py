import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

import matplotlib.cm

font = {'family': 'serif',
        'size': 16}
plt.rc('font', **font)

def plot_se_f1(se_p_list, ns_p_list, colors, num_datasets, save_paths, n_i, label_set_map, sep=False):

    se_avgs, se_err_bars = compute_avg(se_p_list)
    f1_avgs, f1_err_bars = compute_avg(ns_p_list)

    plt_list = [plt.subplots() for i in range(num_datasets)]

    for idx in range(num_datasets):
        label_set_id = label_set_map[idx]
        xs = []
        ys = []
        es = []
        s_a_d = se_avgs[idx]
        f_a_d = f1_avgs[idx]
        f_e_d = f1_err_bars[idx]
        r = compute_corr_coef_by_idx(se_p_list[idx], ns_p_list[idx])
        print(f"data_set nr. {label_set_map[idx]} information: ")
        print(f"pearson coef. w.r.t. all data: {r}")
        for k in f_a_d.keys():
            xs.append(s_a_d[k])
            ys.append(f_a_d[k])
            es.append(f_e_d[k])

        if sep:
            plt_list[idx][1].errorbar(xs, ys, es, fmt=" ", color="gray", alpha=0.5)
            label_str = f'label set {label_set_id}'
            scatter_plot_ns_f1(xs, ys, plt_list[idx][1], colors[idx], label_str=label_str,
                               x_label_text="expected noise model error", sep=sep)
            plot_title = f'label set  {label_set_id}'
            plt_list[idx][1].set_title(plot_title)
            r = stats.pearsonr(xs, ys)
            print(f"pearson coef. w.r.t. avg data: {r}")
            print(f"{len(xs)} data points plotted")
        else:
            plt_list[0][1].errorbar(xs, ys, es, fmt=" ", color="gray", alpha=0.5)
            label_str = f'label set {label_set_id}'
            scatter_plot_ns_f1(xs, ys, plt_list[0][1], colors[idx], label_str=label_str,
                               x_label_text="expected noise model error", sep=sep)
            label_set_map_str = [str(i) for i in label_set_map]
            plot_title = f"label set  {','.join(label_set_map_str)}"
            r = stats.pearsonr(xs, ys)
            print(f"pearson coef. w.r.t. avg data: {r}")
            print(f"{len(xs)} data points plotted")

    save_plots(plt_list, save_paths, sep)


def plot_ns_f1(performance_list, colors, num_datasets, save_paths, n_i, label_set_map, sep=False):

    if n_i:
        x_label_text = "sample size $n_i$"
    else:
        x_label_text = "sample size n"

    f1_avgs, f1_err_bars = compute_avg(performance_list)

    plt_list = [plt.subplots() for i in range(num_datasets)]

    for idx in range(num_datasets):
        ns, err_bar_list = extract_sorted_ns(f1_err_bars[idx])
        ns, f1_avg_list = extract_sorted_ns(f1_avgs[idx])
        label_set_id = label_set_map[idx]

        if sep:
            plt_list[idx][1].errorbar(ns, f1_avg_list, err_bar_list, fmt=" ", color="gray", alpha=0.5)
            label_str = f'label set {label_set_id}'
            scatter_plot_ns_f1(ns, f1_avg_list, plt_list[idx][1], colors[idx], label_str,
                               x_label_text=x_label_text)
            plot_title = f'label set  {label_set_id}'
            plt_list[idx][1].set_title(plot_title)
        else:
            plt_list[0][1].errorbar(ns, f1_avg_list, err_bar_list, fmt=" ", color="gray", alpha=0.5)
            label_str = f'label set {label_set_id}'
            scatter_plot_ns_f1(ns, f1_avg_list, plt_list[0][1], colors[idx], label_str,
                               x_label_text=x_label_text)
            label_set_map_str = [str(i) for i in label_set_map]
            plot_title = f"label set  {','.join(label_set_map_str)}"
            plt_list[0][1].set_title(plot_title)

        print(f"{len(ns)} data points plotted")

    save_plots(plt_list, save_paths, sep, legend_loc=4)




def plot_ns_f1_w_wo_noise_handling(f1_wn, f1_won, colors, label_set_map, save_paths, n_i, sep=False):
    assert not sep

    if n_i:
        x_label_text = "sample size $n_i$"
    else:
        x_label_text = "sample size n"

    f1_avgs, f1_err_bars = compute_avg([f1_wn, f1_won])
    f1_wn_avg, f1_wn_err = f1_avgs[0], f1_err_bars[0]
    f1_won_avg, f1_won_err = f1_avgs[1], f1_err_bars[1]

    fig, axs = plt.subplots()

    ns1, err_bar_list_wn = extract_sorted_ns(f1_wn_err)
    ns2, f1_avg_list_wn = extract_sorted_ns(f1_wn_avg)

    ns3, err_bar_list_won = extract_sorted_ns(f1_won_err)
    ns4, f1_avg_list_won = extract_sorted_ns(f1_won_avg)

    assert ns1 == ns2 == ns3 == ns4
    ns = ns1

    wn_label_str = 'with noise handling'
    won_label_str = 'w/o noise handling'

    axs.errorbar(ns1, f1_avg_list_wn, err_bar_list_wn, fmt=" ", color="gray", alpha=0.5)
    scatter_plot_ns_f1(ns1, f1_avg_list_wn, axs, colors[0], wn_label_str,
                       x_label_text=x_label_text)

    axs.errorbar(ns2, f1_avg_list_won, err_bar_list_won, fmt=" ", color="gray", alpha=0.5)
    scatter_plot_ns_f1(ns2, f1_avg_list_won, axs, colors[1], won_label_str,
                       x_label_text=x_label_text, fmt='s')

    plot_title = f'label set {label_set_map[0]}'
    axs.set_title(plot_title)

    print(f"{len(ns1)} data points plotted")
    save_plots([(fig, axs)], save_paths, sep, legend_loc=4)







def scatter_plot_ns_f1(x_list, y_list, axes_obj, color, label_str,
                       x_label_text, fmt="o", sep=False):
    axes_obj.plot(x_list, y_list, marker=fmt, ls='None', color=color, alpha=1, label=label_str)
    # axes_obj.plot(x_list, y_list, marker=fmt, color=color, alpha=1)
    axes_obj.set_xlabel(x_label_text)
    axes_obj.set_ylabel('F1 score')
    return axes_obj


def save_plots(plt_list, save_paths, sep, legend_loc=1):
    if sep:
        for i, plot_tuple in enumerate(plt_list):
            plot_tuple[0].savefig(save_paths[i], bbox_inches='tight', format='pdf')
            plt.close(plot_tuple[0])
    else:
        plt_list[0][1].legend(loc=legend_loc, prop={'size': 10})
        plt_list[0][0].savefig(save_paths[0], bbox_inches='tight', format='pdf')
        for i, plot_tuple in enumerate(plt_list):
            plt.close(plot_tuple[0])


def compute_avg(input_list):
    """
    :param input_list: list consisting of dicts
    :return: a list of dicts. Each dict contains the average value/std of a experiment
    """

    avg_list = []
    err_list = []
    for acc_hist in input_list:
        summary_dict = dict()
        avg_dict = dict()
        err_dict = dict()
        for acc_dict in acc_hist:
            for k, v in acc_dict.items():
                if k not in summary_dict:
                    summary_dict[k] = []
                else:
                    summary_dict[k].append(v)
        for k, v in summary_dict.items():
            avg_dict[k] = np.average(v)
            err_dict[k] = np.std(v)

        avg_list.append(avg_dict)
        err_list.append(err_dict)

    return avg_list, err_list

def get_dict_by_idx(d, idx):
    # in the Estonian dataset, we have three noisy label sets, identified by idx
    # this function extracts all results related to the noisy label with idx
    output = dict()
    for k in d.keys():
        if k[1] == idx:
            output[k[0]] = d[k]
    return output



def extract_sorted_ns(input_dict):
    ns = []
    ns = list(input_dict.keys())
    ns.sort()
    return ns, [input_dict[i] for i in ns]

def compute_corr_coef_by_idx(se_list_raw, f1_list_raw):
    data_x, data_y = [], []
    for s, f in zip(se_list_raw, f1_list_raw):
        if s is None:
            break
        for k in s.keys():
            data_x.append(s[k])
            data_y.append(f[k])

    return stats.pearsonr(data_x, data_y)



def gen_output_paths(output_root, base_string, label_set_map, sep=False):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if sep:
        output_paths = [os.path.join(output_root, f"{base_string}_{l}.pdf") for l in label_set_map]
    else:
        # we still create n paths
        output_paths = [os.path.join(output_root, f"{base_string}_allsets.pdf") for l in label_set_map]
    return output_paths