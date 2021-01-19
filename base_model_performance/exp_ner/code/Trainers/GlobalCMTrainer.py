import torch
from torch.utils import data
import numpy as np
import logging
from ner_datacode import Evaluation

from Trainers.Trainer import Trainer
from models import create_base_model, Global_CM
from dataset import CoNllDataset



class GlobalCMTrainer(Trainer):
    def __init__(self, all_datasets, train_clean, train_noisy, num_labels, embedding_vector_size,
                 label_representation, SETTINGS, init_mat, summary_writer, device):
        super(GlobalCMTrainer, self).__init__(all_datasets, train_clean, train_noisy, num_labels, embedding_vector_size,
                                          label_representation, SETTINGS, summary_writer, device)
        self.init_mat = init_mat
        self.fix_transition = SETTINGS["FIX_TRANSITION"]

    def train(self):
        # create the based model and the global model
        base_model, feature_extractor = create_base_model(self.embedding_vector_size,
                                                         self.SETTINGS["LSTM_SIZE"], self.SETTINGS["DENSE_SIZE"],
                                                         self.SETTINGS["DENSE_ACTIVATION"],
                                                         self.label_representation.get_num_labels())
        base_model = base_model.to(self.device)
        base_loss = torch.nn.CrossEntropyLoss()
        base_optimizer = torch.optim.Adam(base_model.parameters(), lr=self.lr)

        c_loader, channel_weights = self.get_clean_loader_and_init_mat()
        cm_model = Global_CM(base_model, channel_weights, self.fix_transition, self.device)
        cm_modell_loss = torch.nn.NLLLoss()
        cm_model_optimizer = torch.optim.Adam(cm_model.parameters(), lr=self.lr)
        cm_model = cm_model.to(self.device)
        assert self.use_noisy
        self.noise_model = cm_model

        for epoch in range(self.epochs):
            self.train_epoch(base_model, c_loader, base_loss, base_optimizer, self.device, comment='clean')
            n_loader = self.prepare_noisy_dataloader(self.train_noisy_full, None,
                                                     len(self.train_clean) * self.SETTINGS["NOISE_FACTOR"])
            self.train_epoch(cm_model, n_loader, cm_modell_loss, cm_model_optimizer, self.device,
                             comment='noisy')
            if epoch % self.report_interval == 0:
                self.evaluate(base_model, epoch)

        logging.info(self.long_test_evaluation)
        return Evaluation.extract_f_score(self.long_test_evaluation)

    def get_clean_loader_and_init_mat(self):
        """
        :return: a dataloader for the clean subset and a noise matr used for matrix initialization
        """

        c_dataset = CoNllDataset(self.train_clean)
        c_loader = data.DataLoader(c_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        if self.SETTINGS["USE_IDENTITY_MATRIX"]:
            noise_matrix = np.eye(self.label_representation.get_num_labels(),
                                  self.label_representation.get_num_labels())
        else:
            noise_matrix = self.init_mat

        return c_loader, noise_matrix
