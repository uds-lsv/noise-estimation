from Experimenters.Experimenter import Experimenter
from Trainers.BaseTrainer import BaseTrainer
from Trainers.GlobalCMTrainer import GlobalCMTrainer
import theoretical_var_ni
import expected_se
import utils


# experiments that study the relationship between Accuracy (A) /F1 Score and number of clean Samples (S)
class SAExperimenter(Experimenter):
    def __init__(self, all_datasets, label_representation, test_label_representation, embedding_vector_size,
                 EXP_SETTINGS, TRAIN_SETTINGS, log_dir, logger, device):
        super(SAExperimenter, self).__init__(all_datasets, label_representation, test_label_representation,
                                             embedding_vector_size, EXP_SETTINGS, TRAIN_SETTINGS, log_dir,
                                             logger, device)

    def run_exp(self):
        mat_est_dict, f1_dict = self.init_dict_and_list()
        for nt in range(self.num_times):  # loop over number of runs
            for i, n in enumerate(self.ns):  # loop over different number of clean samples
                train_clean, train_noisy = self.get_train_subsets(n, self.train_clean_full,
                                                                  self.train_noisy_full)

                noise_mat = utils.compute_noise_matrix(train_clean, train_noisy, self.label_representation)

                # estimate theo. noise_matrix estimation error
                if self.uniform_sampling:
                    probability_yis = theoretical_var_ni.estimate_probability_yis(self.train_clean_full,
                                                                                  len(self.train_noisy_full))
                    mat_est_dict[n] = \
                        theoretical_var_ni.expected_se(n, self.true_noise_matrix, probability_yis)
                else:
                    mat_est_dict[n] = \
                        expected_se.expected_se_same_nis(n, self.true_noise_matrix)

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
