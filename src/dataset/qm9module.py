from .qm9_m import QM9M
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import torch


class QM9DataModule(pl.LightningDataModule):
    def __init__(self, trainconf, fold_idx=0):
        super().__init__()
        self.batch_size = trainconf.batch_size
        self.num_workers = trainconf.num_workers
        self.data_path = trainconf.data_dir
        self.fold_idx = fold_idx

    def setup(self, stage):
        qm9m = QM9M(
            root=self.data_path + f"/qm9m",
            excluded_testest=self.data_path + f'/split_{self.fold_idx}/data_test.txt'
            # dataset_path= self.data_path + f'/split_{self.fold_idx}/data_train.txt'
        )

        self.train_dataset, self.val_dataset = train_test_split(
            qm9m, test_size=0.2, random_state=42
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
