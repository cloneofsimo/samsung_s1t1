from .s1t1_sdf import SDFDataset
from torch_geometric.data import DataLoader
import pytorch_lightning as pl


class SDFDataModule(pl.LightningDataModule):
    def __init__(self, trainconf, fold_idx=0):
        super().__init__()
        self.batch_size = trainconf.batch_size
        self.num_workers = trainconf.num_workers
        self.data_path = trainconf.data_dir
        self.fold_idx = fold_idx

    def setup(self, stage):
        self.train_dataset = SDFDataset(
            dataset_path=self.data_path + f"/split_sdf_{self.fold_idx}/data_train.txt",
            sdf_path=self.data_path + f"/train_sdf/",
            train=True,
        )

        self.val_dataset = SDFDataset(
            dataset_path=self.data_path + f"/split_sdf_{self.fold_idx}/data_val.txt",
            sdf_path=self.data_path + f"/train_sdf/",
            train=False,
        )

        self.test_dataset = SDFDataset(
            dataset_path=self.data_path + f"/split_{self.fold_idx}/data_test.txt",
            sdf_path=self.data_path + f"/train_sdf/",
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
