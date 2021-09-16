import torch
import torch.nn as nn

import pytorch_lightning as pl
import numpy as np
from ..core.gine import GINE
from ..core.gat import GAT
from ..core.gatedgcn import GatedGCN
from ..core.pna import PNA
from ..core.gine_emb import GINEEmb
from ..core.gcn_emb import GCNEmb
from ..core.gat_emb import GATEmb

criterions = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "sl1": nn.SmoothL1Loss(beta=0.002),
}


models = {"gine": GINE, "gat": GAT, "gatedgcn": GatedGCN, "pna": PNA, "gine_emb" : GINEEmb, "gcn_emb" : GCNEmb, "gat_emb" : GATEmb}

lr_sch = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "step": torch.optim.lr_scheduler.StepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}


class BaselineSupervisedRegressor(pl.LightningModule):
    def __init__(self, optconf, modelconf, criterionconf, n_ydim):
        super().__init__()
        self.model = models[modelconf.name](n_ydim=n_ydim, **modelconf)
        self.criterion = criterions[criterionconf.name]

        self.optconf = optconf

    def forward(self, x):
        return self.model(x.x, x.edge_index, x.edge_attr, x.batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.criterion(out, batch.y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.criterion(out, batch.y)
        return {"loss": loss, "correct": abs(out - batch.y).mean().item()}

    def validation_epoch_end(self, outputs):
        losses = [loss["loss"] for loss in outputs]
        correct = [loss["correct"] for loss in outputs]
        self.log("val_loss", torch.stack(losses).mean(), prog_bar=True)
        self.log("val_mae_loss", np.array(correct).mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optconf.lr)

        lr_scheduler = lr_sch[self.optconf.lr_sch](
            optimizer, mode="min", patience=10, factor=0.5, verbose=True, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }
