import torch
from torch.utils import data
import numpy as np


class CoNllDataset(data.Dataset):
    def __init__(self, instances, onehot=False):
        self.instances = instances
        embs = np.asarray([ins.embedding for ins in instances])
        labels = np.asarray([ins.label_emb for ins in instances])
        self.words = [ins.word for ins in instances]

        # convert to torch
        self.embs = torch.from_numpy(embs).float()
        if onehot:
            self.labels = torch.from_numpy(labels).float()
        else:
            self.labels = torch.from_numpy(labels).argmax(dim=1)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        emb = self.embs[index]
        label = self.labels[index]
        word = self.words[index]
        return emb, label, word