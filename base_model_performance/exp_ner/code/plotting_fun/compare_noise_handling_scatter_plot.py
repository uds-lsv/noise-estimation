import pickle
import argparse
import os
from plotting_fun import plotting_estonian_utils
from pylab import *
import matplotlib.cm



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir_wn', type=str, required=True,
                        help='directory of experiment with noise handling')
    parser.add_argument('--exp_dir_won', type=str, required=True,
                        help='directory of experiment without noise handling')
    parser.add_argument('--plot_name_prefix', type=str, default='w_and_wo_noise_handling_ls')
    parser.add_argument('--ni', action='store_true',
                        help='Does the noise handling experiment use Fixed Sampling? ni means Fixed Sampling')
    parser.add_argument('--label_set_id', type=int, required=True,
                        help='label set id')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # some sanity checks
    if args.ni:
        assert '_ni_' in args.exp_dir_wn
        assert 'ls' + str(args.label_set_id) + '_' in args.exp_dir_wn


    performance_wn_file = os.path.join(args.exp_dir_wn, "plots/plot_data/acc_hist.pkl")
    performance_won_file = os.path.join(args.exp_dir_won, "plots/plot_data/acc_hist.pkl")

    output_dir = args.output_dir
    cmap = matplotlib.cm.get_cmap("Paired").colors
    # colors = [cmap[7], cmap[0]]
    colors = ['#ff8c00', '#00bfff']  # to match the color of the first version of the code
    #
    label_set_map = [args.label_set_id]

    # with open(mat_se_error, 'rb') as file:
    #     se_list = pickle.load(file)

    with open(performance_wn_file, 'rb') as file:
        f1_wn = pickle.load(file)

    with open(performance_won_file, 'rb') as file:
        f1_won = pickle.load(file)

    suffix = 'ni' if args.ni else 'n'
    suffix += 'fixC' if 'fixC' in args.exp_dir_wn else ''

    output_paths = plotting_estonian_utils.gen_output_paths(output_dir, f'{args.plot_name_prefix}_{suffix}',
                                                            label_set_map,
                                                            sep=True)  # set sep=True although we don't sep
    plotting_estonian_utils.plot_ns_f1_w_wo_noise_handling(f1_wn, f1_won,
                                                           colors, label_set_map, output_paths, args.ni, sep=False)


if __name__ == "__main__":
   main()




