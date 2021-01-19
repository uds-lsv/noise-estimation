import pickle
import argparse
import os
from plotting_fun import plotting_estonian_utils as p_utils
from pylab import *
import matplotlib.cm


parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir_fix', type=str, required=True,
                    help='directory of experiment Fixed Sampling')
parser.add_argument('--exp_dir_var', type=str, required=True,
                    help='directory of experiment Variable Sampling')
parser.add_argument('--plot_name_prefix', type=str, default='fix_vs_var_ls')
parser.add_argument('--label_set_id', type=int, required=True,
                    help='label set id')
parser.add_argument('--sub_sample_ns', type=int, nargs='+', required=True,
                    help='only show some ns on the boxplot to save space')
parser.add_argument('--sub_sample_n', type=int, nargs='+', required=True,
                    help='only show some n on the boxplot to save space')
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

# some sanity checks
assert '_ni_' in args.exp_dir_fix
assert '_n_' in args.exp_dir_var
assert 'ls'+str(args.label_set_id)+'_' in args.exp_dir_fix
assert 'ls'+str(args.label_set_id)+'_' in args.exp_dir_var
c = args.sub_sample_n[0] / args.sub_sample_ns[0]
for idx, v in enumerate(args.sub_sample_ns):
    assert v*c == args.sub_sample_n[idx]


def main():
    performance_fix_file = os.path.join(args.exp_dir_fix, "plots/plot_data/acc_hist.pkl")
    performance_var_flie = os.path.join(args.exp_dir_var, "plots/plot_data/acc_hist.pkl")

    output_dir = args.output_dir


    with open(performance_fix_file, 'rb') as file:
        f1_fix = pickle.load(file)

    with open(performance_var_flie, 'rb') as file:
        f1_var = pickle.load(file)

    cmap = matplotlib.cm.get_cmap("Paired").colors
    colors = [cmap[1], cmap[3]]
    label_set_map = [args.label_set_id]

    plot_boxplot(f1_fix, f1_var, colors, args)

def plot_boxplot(fix_data, var_data, colors, args):
    output_dir = args.output_dir
    # box_plot
    n_groups = len(args.sub_sample_ns)
    bar_width = 0.35
    opacity = 0.8
    label_maps = {0: 'Variable Sampling', 1: 'Fixed Sampling'}
    num_of_bars = 2
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    f1_data_raw_list = []

    min_y, max_y = 999, 0
    subset_ids = [args.sub_sample_n, args.sub_sample_ns]
    f1_avg_subs = []
    for i, f1_raw in enumerate([var_data, fix_data]):
        f1_sub = get_sub_ns_dict(f1_raw, subset_ids[i])
        f1_avg_list, f1_err_bar_list = p_utils.compute_avg([f1_sub])
        f1_avg_subs.append(f1_avg_list[0])
        f1_err_bar_dict, f1_avg_dict = f1_err_bar_list[0], f1_avg_list[0]
        min_y = update_min_y(min_y, list(f1_avg_dict.values()), list(f1_err_bar_dict.values()))
        max_y = update_max_y(max_y, list(f1_avg_dict.values()), list(f1_err_bar_dict.values()))
        assert len(f1_avg_dict) == n_groups

        ax.bar(index + i * bar_width, f1_avg_dict.values(), bar_width,
               alpha=opacity,
               color=colors[i],
               label=label_maps[i])
        # ax.errorbar(index + i * bar_width, f1_list.values(), f1_err_bar_list.values(), capsize=5, fmt=" ", color="gray", alpha=0.5)
        ax.errorbar(index + i * bar_width, f1_avg_dict.values(), f1_err_bar_dict.values(), fmt=" ", color="gray",
                    alpha=0.5)

    ax.set_xlabel('sample size $n$/$n_i$')
    ax.set_ylabel('F1 Score')
    y_lim_lower, y_lim_upper = nearest_multiple(5, min_y, higher=False, offset=5), nearest_multiple(5, max_y,
                                                                                                    higher=True,
                                                                                                offset=0)
    plot_title = f"label set  {args.label_set_id}"
    ax.set_title(plot_title)
    ax.set_ylim([y_lim_lower, y_lim_upper])
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([f'{k1}/{k2}' for k1, k2 in zip(f1_avg_subs[0].keys(), f1_avg_subs[1].keys())])
    ax.legend(loc=4, prop={'size': 10})

    suffix = '_fixC' if 'fixC' in args.exp_dir_fix else ''



    output_path = os.path.join(args.output_dir, f'{args.plot_name_prefix}{suffix}_{args.label_set_id}.pdf')
    fig.savefig(output_path, bbox_inches='tight', format='pdf')
    plt.close(fig)




def get_sub_ns_dict(f1_raw, sub_sample_ns):
    f1_sub = []
    for idx, f1_dict in enumerate(f1_raw):
        f1_dict_sub = {}
        for k, v in f1_dict.items():
            if k in sub_sample_ns:
                f1_dict_sub[k] = v
        f1_sub.append(f1_dict_sub)

    return f1_sub



def update_min_y(cur_min_y, score_list, err_list):
    min_score = np.min(np.array(score_list) - np.array(err_list))
    if cur_min_y > min_score:
        return min_score
    else:
        return cur_min_y

def update_max_y(cur_max_y, score_list, err_list):
    max_score = np.max(np.array(score_list) + np.array(err_list))
    if cur_max_y > max_score:
        return cur_max_y
    else:
        return max_score

def nearest_multiple(base, value, higher=True, offset=5):
    if higher:
        return base * np.ceil(value/base) + offset
    else:
        return base * np.floor(value/base) - offset





if __name__ == "__main__":
   main()

# for label_idx in [0, 1, 2]:
#     output_path = os.path.join(output_root, f'label_set_{label_set_map[label_idx]}.pdf')
#     plot_boxplot(label_idx, output_path)


