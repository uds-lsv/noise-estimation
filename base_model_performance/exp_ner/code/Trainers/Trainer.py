import numpy as np
import logging
from utils import create_subset, simple_evaluation, long_evaluation
from dataset import CoNllDataset
from torch.utils import data
from scipy import special

class Trainer:
    def __init__(self, all_datasets, train_clean, train_noisy, num_labels, embedding_vector_size,
                 label_representation, SETTINGS, summary_writer, device):
        self.train_clean_full = all_datasets[0]
        self.train_noisy_full = all_datasets[1]
        self.train_clean = train_clean
        self.train_noisy = train_noisy
        self.dev = all_datasets[2]
        self.test = all_datasets[3]
        self.num_labels = num_labels
        self.SETTINGS = SETTINGS
        self.sampling_state = np.random.RandomState(SETTINGS["SAMPLE_SEED"])
        self.epochs = SETTINGS["EPOCHS"]
        self.batch_size = SETTINGS["BATCH_SIZE"]
        self.num_workers = SETTINGS["NUM_WORKERS"]
        self.use_identity = SETTINGS["USE_IDENTITY_MATRIX"]
        self.use_noisy = SETTINGS["USE_NOISY"]
        self.report_interval = SETTINGS["REPORT_INTERVAL"]
        self.fix_transition = SETTINGS["FIX_TRANSITION"]
        self.lr = SETTINGS["LEARNING_RATE"]
        self.embedding_vector_size = embedding_vector_size
        self.label_representation = label_representation

        # evaluation related
        self.best_dev = -1
        self.long_test_evaluation = None

        self.sw = summary_writer
        self.device = device

        self.dev_dataset = CoNllDataset(self.dev)
        self.dev_dataloader = data.DataLoader(self.dev_dataset, batch_size=SETTINGS["BATCH_SIZE"], shuffle=False,
                                              num_workers=0)
        self.test_dataset = CoNllDataset(self.test)
        self.test_dataloader = data.DataLoader(self.test_dataset, batch_size=SETTINGS["BATCH_SIZE"], shuffle=False,
                                               num_workers=0)

        # some placeholders, will be overwritten by subclasses
        self.noise_model = None
        self.init_mat = None

    def prepare_noisy_dataloader(self, train_noisy_full, cs, size):
        subset, _ = create_subset(train_noisy_full, cs, size, self.sampling_state, sequential=False)
        dataset = CoNllDataset(subset)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader

    def train(self):
        # do not implement, it should be an abstract method
        raise NotImplementedError("Please Implement this method")

    def train_epoch(self, model, dataloader, loss_fn, optimizer, device, comment):
        model.train()
        for data in dataloader:
            xs, labels = data[0], data[1]
            xs = xs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            y_pred = model(xs)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

    def evaluate(self, model, epoch):
        eval_dev = simple_evaluation(model, self.dev, self.dev_dataloader, self.device,
                                     self.label_representation, self.SETTINGS)
        eval_test = simple_evaluation(model, self.test, self.test_dataloader, self.device,
                                      self.label_representation, self.SETTINGS)
        logging.info(f'Epoch {epoch + 1}\tCurrent F1 for DEV: {eval_dev}\tTEST: {eval_test}')

        if eval_dev > self.best_dev:
            self.best_dev = eval_dev
            self.long_test_evaluation = long_evaluation(model, self.test, self.test_dataloader, self.device,
                                                         self.label_representation, self.SETTINGS)

    def get_trainer_info(self):
        info = "Trainer information: \n"
        info += f"input matrix: \n" \
            f"{np.around(self.init_mat, 2)}\n" \
            f"final matirx: \n"

        assert self.use_noisy
        if self.SETTINGS["MODEL_TYPE"] == "GLOBAL_CM" :
            if self.fix_transition:
                info += f"transition matrix fixed\n"
            else:
                mat = self.convert_to_prob_mat(self.noise_model.transition_mat.data.cpu().numpy())
                info += f"{mat}\n"

        return info

    def convert_to_prob_mat(self, mat):
        mat_after_softmax = special.softmax(mat, axis=1)
        return np.around(mat_after_softmax, 2)
