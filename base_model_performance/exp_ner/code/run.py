import torch
import datetime
import utils
from Experimenters.SAExperimenter import SAExperimenter
from Experimenters.SAMExperimenter import SAMExperimenter
from experimentalsettings import ExperimentalSettings
import argparse
import logging


def run_exps():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument('--input_dir', type=str, required=True,
                        help='dir that contains the dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='root directory that contains images')
    parser.add_argument('--word_embedding_path', type=str, default="../data/fasttext/cc.et.300.bin",
                        help='FastText embedding path')

    # data reading
    parser.add_argument('--data_separator', type=str, default="\t",
                        help='input format: token[data_separator]label')
    parser.add_argument('--label_format', default='io', choices=['io', 'bio'],
                        help='io or bio format')


    # experiment related
    parser.add_argument('--label_set', type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--num_times', type=int, required=True,
                        help='triple of (start stop step_size)')
    parser.add_argument('--ns', nargs='+', type=int, default=[], help='number of samples for each label')
    parser.add_argument('--uniform_sampling', action='store_true',
                        help='if yes, then we sample the clean data randomly, '
                             'otherwise we sample the equal size of data points per label')
    parser.add_argument('--exp_settings', type=str, required=True, help='a list of experiment settings')
    parser.add_argument('--train_settings', type=str, required=True, help=' a list training settings')
    parser.add_argument('--random_seed', type=int, required=True, help='random_seed')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')

    # Training related arguments
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch_size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning_rate')

    args = parser.parse_args()

    # read experiment settings/configs
    exp_setting_name = args.exp_settings
    train_setting_name = args.train_settings

    all_datasets, label_representation, test_label_representation, embedding_vector_size =\
        utils.load_all_processed_data(args)


    staring_time = datetime.datetime.now()
    staring_time_str = staring_time.strftime("%b_%d_%H_%M_%S")

    logger, log_dir = utils.create_logger(args.output_dir, args, staring_time)

    logger.info(f"EXP={exp_setting_name}, TRAIN={train_setting_name} begins")

    EXP_SETTINGS = ExperimentalSettings.load_json(exp_setting_name, logger, dir_path="./exp_config/")
    EXP_SETTINGS["LABEL_SET"] = args.label_set
    EXP_SETTINGS["OUTPUT_DIR"] = args.output_dir
    if args.ns != []:
        EXP_SETTINGS["NS"] = args.ns

    TRAIN_SETTINGS = ExperimentalSettings.load_json(train_setting_name, logger)

    assert label_representation.get_num_labels() == EXP_SETTINGS["NUM_LABELS"]

    if 'fixC' in exp_setting_name:
        experimenter = SAMExperimenter(all_datasets, label_representation, test_label_representation,
                                       embedding_vector_size, EXP_SETTINGS, TRAIN_SETTINGS, log_dir, logger,
                                       device)
    else:
        experimenter = SAExperimenter(all_datasets, label_representation, test_label_representation,
                                      embedding_vector_size, EXP_SETTINGS, TRAIN_SETTINGS, log_dir, logger,
                                      device)

    experimenter.run_exp()

    end_time = datetime.datetime.now()
    time_diff = end_time - staring_time
    total_time = time_diff.total_seconds()/60
    logger.info(f"EXP={exp_setting_name}, TRAIN={train_setting_name} finished, total_time={total_time}mins")


if __name__ == "__main__":
   run_exps()