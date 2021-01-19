from Experimenters.Experimenter import Experimenter
from Trainers.GlobalCMTrainer import GlobalCMTrainer
from Trainers.BaseTrainer import BaseTrainer
import theoretical_var_ni
import expected_se
import utils
import copy


# experiments that study the relationship between Accuracy (A) /F1 Score and number of clean Samples (S)
# We provide noise matrices of different qualities, but use the SAME clean samples to train
class SAMExperimenter(Experimenter):
    def __init__(self, all_datasets, label_representation, test_label_representation, embedding_vector_size,
                 EXP_SETTINGS, TRAIN_SETTINGS, log_dir, logger, device):
        super(SAMExperimenter, self).__init__(all_datasets, label_representation, test_label_representation,
                                              embedding_vector_size, EXP_SETTINGS, TRAIN_SETTINGS, log_dir,
                                              logger, device)

        # We fix the clean sub set for training. But we initialize the global model
        # using transition matrices of different qualities
        self.fix_c_size = EXP_SETTINGS["FIX_C_SIZE"]

    def run_exp(self):
        mat_est_dict, f1_dict = self.init_dict_and_list()
        for nt in range(self.num_times):  # loop over number of runs
            for i, n in enumerate(self.ns):  # loop over different number of clean samples
                train_clean_for_mat, train_noisy_for_mat = \
                    self.get_train_subsets(n, self.train_clean_full, self.train_noisy_full)

                # in this experiment, we provide the trainer the noise_mat estimated by different subsets
                noise_mat = utils.compute_noise_matrix(train_clean_for_mat, train_noisy_for_mat,
                                                       self.label_representation)

                if self.uniform_sampling:
                    probability_yis = \
                        theoretical_var_ni.estimate_probability_yis(self.train_clean_full,
                                                                    len(self.true_noise_matrix))
                    mat_est_dict[n] = \
                        theoretical_var_ni.expected_se(n, self.true_noise_matrix, probability_yis)
                else:
                    mat_est_dict[n] = \
                        expected_se.expected_se_same_nis(n, self.true_noise_matrix)

                # however, the trainer will train on a fixed, small subsets
                # overwrite sampled data
                train_clean, train_noisy = self.get_train_subsets(n, self.train_clean_full,
                                                                  self.train_noisy_full)

                training_full = self.all_datasets

                if self.TRAIN_SETTINGS['MODEL_TYPE'] == 'BASE':
                    trainer = BaseTrainer(training_full, train_clean, train_noisy,
                                          self.label_representation.get_num_labels(),
                                          self.embedding_vector_size,
                                          self.label_representation, self.TRAIN_SETTINGS, noise_mat, None,
                                          self.device)
                else:
                    assert self.TRAIN_SETTINGS['MODEL_TYPE'] == 'GLOBAL_CM'
                    trainer = GlobalCMTrainer(training_full, train_clean, train_noisy,
                                              self.label_representation.get_num_labels(),
                                              self.embedding_vector_size,
                                              self.label_representation, self.TRAIN_SETTINGS, noise_mat, None,
                                              self.device)

                f1 = trainer.train()  # train the model. train() returns the final test f1-score
                f1_dict[n] = f1  # log the test f1-score associated with dataset [idx] and #samples [n]

                self.display_information(nt, n, f1_dict, trainer, self.logger)  # display results

            self.update_and_save_hists(nt, mat_est_dict, f1_dict)  # save logs to disk




