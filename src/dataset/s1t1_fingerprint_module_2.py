from .s1t1_fingerprint_2 import FingerprintDataset
from torch_geometric.data import DataLoader
import pytorch_lightning as pl


class FingerprintDataModule(pl.LightningDataModule):
    def __init__(self, trainconf, fold_idx=0):
        super().__init__()
        self.batch_size = trainconf.batch_size
        self.num_workers = trainconf.num_workers
        self.data_path = trainconf.data_dir
        self.fold_idx = fold_idx

    def setup(self, stage):
        self.train_dataset = FingerprintDataset(
            root=self.data_path + f"/data_train_{self.fold_idx}_fp2",
            dataset_path=self.data_path + f"/split_{self.fold_idx}_2/data_train.txt",
        )

        self.val_dataset = FingerprintDataset(
            root=self.data_path + f"/data_val_{self.fold_idx}_fp2",
            dataset_path=self.data_path + f"/split_{self.fold_idx}_2/data_val.txt",
        )

        self.test_dataset = FingerprintDataset(
            root=self.data_path + f"/data_test_fp2",
            dataset_path=self.data_path + f"/split_{self.fold_idx}_2/data_test.txt",
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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
