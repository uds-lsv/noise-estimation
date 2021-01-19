import torch
from torch.utils import data
import numpy as np
import logging
from ner_datacode import Evaluation

from Trainers.Trainer import Trainer
from models import create_base_model
from dataset import CoNllDataset



class BaseTrainer(Trainer):
    def __init__(self, all_datasets, train_clean, train_noisy, num_labels, embedding_vector_size,
                 label_representation, SETTINGS, init_mat, summary_writer, device):
        super(BaseTrainer, self).__init__(all_datasets, train_clean, train_noisy, num_labels, embedding_vector_size,
                                          label_representation, SETTINGS, summary_writer, device)
        self.init_mat = init_mat
        self.fix_transition = SETTINGS["FIX_TRANSITION"]

    def train(self):
        # create base model
        base_model, feature_extractor = create_base_model(self.embedding_vector_size,
                                                         self.SETTINGS["LSTM_SIZE"], self.SETTINGS["DENSE_SIZE"],
                                                         self.SETTINGS["DENSE_ACTIVATION"],
                                                         self.label_representation.get_num_labels())
        base_model = base_model.to(self.device)
        base_loss = torch.nn.CrossEntropyLoss()
        base_optimizer = torch.optim.Adam(base_model.parameters(), lr=self.lr)

        c_dataset = CoNllDataset(self.train_clean)
        c_loader = data.DataLoader(c_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        for epoch in range(self.epochs):
            self.train_epoch(base_model, c_loader, base_loss, base_optimizer, self.device, comment='clean')

            if self.use_noisy:
                n_loader = self.prepare_noisy_dataloader(self.train_noisy_full, None,
                                                         len(self.train_clean) * self.SETTINGS["NOISE_FACTOR"])
                self.train_epoch(base_model, n_loader, base_loss, base_optimizer, self.device,
                                 comment='noisy')

            if epoch % self.report_interval == 0:
                self.evaluate(base_model, epoch)

        logging.info(self.long_test_evaluation)
        return Evaluation.extract_f_score(self.long_test_evaluation)

