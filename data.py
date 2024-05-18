import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FV
import torch.optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from tqdm import tqdm
import os


import lightning as L
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchmetrics.image.fid import FrechetInceptionDistance
import optuna
import pickle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import EarlyStopping
class SequenceDataset(Dataset):
    def __init__(self, data, has_labels=True):
        self.data = data
        self.has_labels = has_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.has_labels:
            sequence, label = self.data[idx]
            return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        else:
            sequence = self.data[idx]
            return torch.tensor(sequence, dtype=torch.float32)

class SequenceDataModule(L.LightningDataModule):
    def __init__(self, train_file, test_file, workers, batch_size=32, val_split=0.2):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.val_split = val_split
        self.workers = workers

    def setup(self, stage=None):
        with open(self.train_file, 'rb') as f:
            train_data = pickle.load(f)
        with open(self.test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        train_size = int((1 - self.val_split) * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, val_size])
        self.train_dataset = SequenceDataset(train_data)
        self.val_dataset = SequenceDataset(val_data)
        self.test_dataset = SequenceDataset(test_data, has_labels=False)

    def collate_fn(self, batch):
        if isinstance(batch[0], tuple):
            sequences, labels = zip(*batch)
            sequences = [seq.clone().detach().unsqueeze(-1) for seq in sequences]  # Add extra dimension
            lengths = torch.tensor([len(seq) for seq in sequences])
            sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
            return sequences_padded, lengths, torch.tensor(labels, dtype=torch.long)
        else:
            sequences = [seq.clone().detach().unsqueeze(-1) for seq in batch]  # Add extra dimension
            lengths = torch.tensor([len(seq) for seq in sequences])
            sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
            return sequences_padded, lengths

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.workers)
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.workers)