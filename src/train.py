import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GraphConv, ChebConv
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool

import pytorch_lightning as pl
import numpy as np

import sys
sys.path.insert(0, '..')

# Model Init
class ResGraphModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResGraphModule, self).__init__()

        self.conv = GINEConv(nn.Linear(in_channels, out_channels), eps=1e-5)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x_ = x
        x = self.conv(x, edge_index, edge_attr)
        x = self.relu(x) + x_
        return x


class GNN(nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)

        self.vert_emb = nn.Linear(13, 2 * hidden_channels, bias=False)
        self.edge_emb = nn.Linear(4, 2 * hidden_channels, bias=False)

        self.main = gnn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    ResGraphModule(2 * hidden_channels, 2 * hidden_channels),
                    "x, edge_index, edge_attr -> x",
                ),
                *[
                    (
                        ResGraphModule(2 * hidden_channels, 2 * hidden_channels),
                        "x, edge_index, edge_attr -> x",
                    )
                    for _ in range(8)
                ],
            ],
        )

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_channels, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.vert_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        x = self.main(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        # x = F.dropout(x, p=0.5, training=self.training)

        x = self.head(x)

        return x

class GNNRegressor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GNN(hidden_channels=32)
        self.criterion = nn.SmoothL1Loss(beta=0.002)

    def forward(self, x):
        return self.model(
            x.x, x.edge_index, x.edge_attr, x.batch
        )
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.criterion(out, batch.y)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.criterion(out, batch.y)
        return {'loss': loss, 'correct': abs(out - batch.y).mean().item()}
    
    def validation_epoch_end(self, outputs):
        losses = [loss['loss'] for loss in outputs]
        correct = [loss['correct'] for loss in outputs]
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
        self.log('val_mae_loss', np.array(correct).mean(), prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)
        return optimizer

# DataLoader Init
from mol_dataset import BaselineDataset
from torch_geometric.data import DataLoader

class ChemDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage):
        self.train_dataset = BaselineDataset(
            root="../data_train",
            dataset_path="../data/split_0/data_train.txt",
        )

        self.val_dataset = BaselineDataset(
            root="../data_test",
            dataset_path="../data/split_0/data_val.txt",
        )

        self.test_dataset = BaselineDataset(
            root="../data_inf",
            dataset_path="../data/split_0/data_test.txt",
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

# Training
import argparse
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='val_mae_loss',
    dirpath='checkpoints',
    filename='sample-gnn-{epoch:02d}-{val_mae_loss:.2f}'
)

parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
model = GNNRegressor()
dm = ChemDataModule()

trainer.fit(model, dm)