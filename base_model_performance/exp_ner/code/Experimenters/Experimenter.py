import os
import numpy as np
import json
import pickle
import utils
import copy


class Experimenter():
    def __init__(self, all_datasets, label_representation, test_label_representation, embedding_vector_size,
                 EXP_SETTINGS, TRAIN_SETTINGS, log_dir, logger, device):

        # information needed for training and experiments
        self.all_datasets = all_datasets
        self.label_representation = label_representation
        self.test_label_representation = test_label_representation
        self.embedding_vector_size = embedding_vector_size
        self.EXP_SETTINGS = EXP_SETTINGS
        self.TRAIN_SETTINGS = TRAIN_SETTINGS
        self.label_set = EXP_SETTINGS["LABEL_SET"]
        self.num_times = EXP_SETTINGS["NUM_TIMES"]
        self.ns = EXP_SETTINGS["NS"]
        self.unify_scale = EXP_SETTINGS["UNIFY_SCALE"]
        self.uniform_sampling = EXP_SETTINGS["UNIFORM_SAMPLING"]
        self.random_state = np.random.RandomState(EXP_SETTINGS["RANDOM_SEED"])
        self.num_labels = EXP_SETTINGS["NUM_LABELS"]

        #  Two lists store results after training, will be initialized by the Subclasses
        self.f1_hist = None
        self.mat_est_hist = None

        #  create all directories/paths used for storing the results
        self.output_root = log_dir
        self.exp_info_root = os.path.join(self.output_root, 'exp_info')
        self.plot_root = os.path.join(self.output_root, 'plots')
        self.plot_data_file_root = os.path.join(self.plot_root, 'plot_data')
        self.exp_info_root = os.path.join(self.output_root, 'exp_info')
        self.exp_info_file = os.path.join(self.exp_info_root, 'exp_info.txt')
        file_dirs = [self.output_root, self.exp_info_root, self.plot_root, self.plot_data_file_root, self.exp_info_root]
        utils.check_existence_and_create(file_dirs)
        self.exp_info_file = os.path.join(self.exp_info_root, 'exp_info.txt')
        exp_set_output_path = os.path.join(self.output_root, 'EXP_SETTINGS.txt')
        training_set_output_path = os.path.join(self.output_root, 'TRAINING_SETTINGS.txt')
        self.log_settings(exp_set_output_path, EXP_SETTINGS)
        self.log_settings(training_set_output_path, TRAIN_SETTINGS)

        #  extract datasets for training and testing
        self.train_clean_full = all_datasets[0]
        self.train_noisy_full = all_datasets[1]
        self.train_clean = None
        self.train_noisy = None
        self.dev = all_datasets[2]
        self.test = all_datasets[3]

        # use the empirically estimated matrix from the the full dataset as the true matrix
        self.true_noise_matrix = self.get_noise_matrix()
        self.device = device
        self.logger = logger

    def get_noise_matrix(self):
        true_noise_matrix = utils.compute_noise_matrix(self.train_clean_full, self.train_noisy_full,
                                                       self.label_representation)
        return true_noise_matrix

    def get_data_with_one_noisy_set(self, idx):
        return [self.all_datasets[0], self.all_datasets[1][idx], self.all_datasets[2], self.all_datasets[3]]

    def create_root_name(self, staring_time):
        """
        create the root directory of the outputs
        :param staring_time: used for root name suffix
        """
        suffix = self.EXP_SETTINGS["NAME"]
        suffix += f"_ls{self.label_set}"

        if self.TRAIN_SETTINGS["MODEL_TYPE"] == "GLOBAL_CM":
            if self.TRAIN_SETTINGS["FIX_TRANSITION"]:
                suffix += '_mfixed'
            else:
                suffix += '_mlearned'
        elif self.TRAIN_SETTINGS["MODEL_TYPE"] == "BASE":
            suffix += 'base'
        else:
            raise NotImplementedError("Please Implement this method")
        output_root = os.path.join(self.EXP_SETTINGS["DIR"], suffix, staring_time)
        return output_root

    def log_settings(self, output_path, settings_dict):
        with open(output_path, 'w') as file:
            file.write(json.dumps(settings_dict.settings))

    def get_train_subsets(self, sample_size, train_clean_full, train_noisy_full):
        """
        sample a subset from full dataset
        :param sample_size: number of samples wanted
        :param train_clean_full: full dataset enquiped with clean labels
        :param train_noisy_full: full dataset enquiped with noisy labels
        :return: two subsets with the same instances but different labels.
        """
        if self.uniform_sampling:
            sampled_indices = self.random_state.choice(len(train_clean_full), sample_size, replace=False)
            assert len(set(sampled_indices)) == len(sampled_indices)
            train_clean = [train_clean_full[i] for i in sampled_indices]
            train_noisy = [train_noisy_full[i] for i in sampled_indices]
        else:
            train_clean, train_noisy = utils.sample_data_same_n_for_dataset(sample_size, train_clean_full,
                                                                            train_noisy_full,
                                                                            self.num_labels,
                                                                            self.random_state)

        return train_clean, train_noisy


    def compute_mat_est_err(self, estimated_mat):
        """
        compute squared error of between the GT noise matrix and the estimated matrix
        :param idx: used for extracting the true noise
        :param estimated_mat: empirically estimated mat using paired clean and noisy datasets
        :return: squared error of two given matrices
        """
        se = np.sum(np.square(self.true_noise_matrix - estimated_mat))
        return se

    def init_dict_and_list(self):
        """
        initialize a list to store the F1-score achieved by the model in every run, and a list to store the theoretical
        matrix estimation error in very run
        :return:
        """
        mat_est_dict = dict()
        f1_dict = dict()
        self.mat_est_hist = [None] * self.num_times
        self.f1_hist = [None] * self.num_times
        return mat_est_dict, f1_dict

    def write_dict_to_file(self, mat_est_hist, acc_hist, file_root):
        """
        save both f1/accuracy and matrix estimation error histories into files
        :param mat_est_hist: matrix estimation error history wanted to be saved
        :param acc_hist:  f1/accuracy history wanted to be saved
        :param file_root: root directory for saving
        """
        mat_est_hist_path = os.path.join(file_root, 'mat_est_hist.pkl')
        acc_hist_path = os.path.join(file_root, 'acc_hist.pkl')

        with open(mat_est_hist_path, "wb") as file:
            pickle.dump(mat_est_hist, file)

        with open(acc_hist_path, "wb") as file:
            pickle.dump(acc_hist, file)

    def update_and_save_hists(self, nt, mat_est_dict, f1_dict):
        """
        :param nt: current  #runs
        :param mat_est_dict: key: [#clean_sample,training_data_idx] value: corresponding theo. matrix estimation error
        :param f1_dict: [#clean_sample,training_data_idx] value: corresponding test f1-score
        :return: None
        """
        assert self.mat_est_hist[nt] is None
        assert self.f1_hist[nt] is None
        self.mat_est_hist[nt] = copy.deepcopy(mat_est_dict)
        self.f1_hist[nt] = copy.deepcopy(f1_dict)

        # will overwrite the plot data from previous iteration as we only want the final result
        self.write_dict_to_file(self.mat_est_hist, self.f1_hist, self.plot_data_file_root)

    def display_information(self, nt, n, f1_dict, trainer, logger):
        info = f"run={nt + 1} size={n} label_set={self.EXP_SETTINGS['LABEL_SET']} " \
               f"cur_f1={f1_dict[n]}"
        info += trainer.get_trainer_info()
        logger.info(info)
        with open(self.exp_info_file, "a") as file:
            file.write(info + '\n')

    def run_exp(self):
        """
        should be implemented in the subclasses
        """
        raise NotImplementedError("Please Implement this method")
