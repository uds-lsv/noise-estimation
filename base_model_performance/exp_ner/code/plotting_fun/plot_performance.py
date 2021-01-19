"""
Given one experiment directory, plot num_clean_samples against performance (F1 Score/Accuracy)
Also support multiple experiment directories. In this case, we support a) plot performance from different experiments on
a single plot for performance comparison. b) plot performance separately, one plot for each experiment.
"""
import os
from plotting_fun import plotting_estonian_utils
from pylab import *
import pickle
import matplotlib.cm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dirs', type=str, nargs='+', required=True,
                    help='dirs of the output of different experiments')
parser.add_argument('--label_set_ids', type=int, nargs='+', required=True,
                    help='ids of the labesets used in your experiments')
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--plot_name_prefix', type=str, default='Estonian_F1')
parser.add_argument('--ni', action='store_true',
                    help='Do the experiments use Fixed Sampling? ni means Fixed Sampling')
parser.add_argument('--store_plots_separately', action='store_true',
                    help='one plot for each experiment? Or one plot for all experiments?')
args = parser.parse_args()

exp_dirs = args.exp_dirs
cmap = matplotlib.cm.get_cmap("Paired").colors
assert len(cmap) >= len(exp_dirs)  # make sure we have enough colors
colors = [cmap[2], cmap[4], cmap[6], cmap[8]]
# colors = [cmap[i] for i in range(len(exp_dirs))]

# some sanity checks
if args.ni:
    for exp_dir in exp_dirs:
        assert '_ni_' in exp_dir
    for l, e in zip(args.label_set_ids, args.exp_dirs):
        assert 'ls'+str(l)+'_' in e
else:
    for exp_dir in exp_dirs:
        assert '_n_' in exp_dir
        assert '_ni_' not in exp_dir
    for l,e in zip(args.label_set_ids, args.exp_dirs):
        assert 'ls'+str(l)+'_' in e

ns_p_paths = [os.path.join(d, "plots/plot_data/acc_hist.pkl") for d in exp_dirs]  # x-axis: ns, y-axis: performance (F1)
ns_p_list = []
for np in ns_p_paths:
    with open(np, 'rb') as file:
        performance = pickle.load(file)
    ns_p_list.append(performance)

se_p_paths = [os.path.join(d, "plots/plot_data/mat_est_hist.pkl") for d in exp_dirs]
se_p_list = []

for np in se_p_paths:
    with open(np, 'rb') as file:
        performance = pickle.load(file)
    se_p_list.append(performance)

label_set_map = args.label_set_ids


suffix = 'ni' if args.ni else 'n'
suffix += '_fixC' if 'fixC' in exp_dirs[0] else ''
output_paths = plotting_estonian_utils.gen_output_paths(args.output_dir, f'{args.plot_name_prefix}_ns_{suffix}', label_set_map,
                                                        sep=args.store_plots_separately)


plotting_estonian_utils.plot_ns_f1(ns_p_list, colors, len(exp_dirs), output_paths, args.ni, label_set_map,
                                   sep=args.store_plots_separately)

output_paths = plotting_estonian_utils.gen_output_paths(args.output_dir, f'{args.plot_name_prefix}_se_{suffix}', label_set_map,
                                                        sep=args.store_plots_separately)

plotting_estonian_utils.plot_se_f1(se_p_list, ns_p_list, colors, len(exp_dirs), output_paths, args.ni, label_set_map,
                                   sep=args.store_plots_separately)



