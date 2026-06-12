import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ERA5ShortTermDataset(Dataset):
    def __init__(self, split='train', data_dir='/path/to/data'):
        self.split = split
        self.data_dir = data_dir
        self.num_samples = 1000 if split == 'train' else 200

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        hist_x = torch.randn(12, 5, 80, 80)  # [T_in, C, H, W]
        future_y = torch.randn(4, 5, 80, 80)  # [T_out, C, H, W]
        month_idx = torch.randint(0, 12, (1,)).item()

        return hist_x, future_y, month_idx


class ERA5DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = ERA5ShortTermDataset(split='train', data_dir=self.data_dir)
        self.val_dataset = ERA5ShortTermDataset(split='val', data_dir=self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)