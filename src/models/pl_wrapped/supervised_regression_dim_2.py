import torch
import torch.nn as nn

import pytorch_lightning as pl
import numpy as np

from models.core.sch_emb import SchEmb
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


models = {
    "gine": GINE,
    "gat": GAT,
    "gatedgcn": GatedGCN,
    "pna": PNA,
    "gine_emb": GINEEmb,
    "gcn_emb": GCNEmb,
    "gat_emb": GATEmb,
    "sch_emb": SchEmb,
}

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
        yh = self.model(x.x, x.edge_index, x.edge_attr, x.batch, x.pos)
        y_aux = yh
        y_gt = yh[:, 0:1] - yh[:, 1:2]

        return {"TARGET": y_gt, "AUX": y_aux}

    def training_step(self, batch, batch_idx):
        out = self(batch)

        # print(batch.y.shape)
        loss = self.criterion(out["TARGET"], batch.y)
        loss_aux = self.criterion(out["AUX"], batch.y_aux) * 0.5
        return {"loss": loss + loss_aux}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.criterion(out["TARGET"], batch.y)
        gap_pred = abs(out["TARGET"])
        gap_gt = abs(batch.y)

        return {"loss": loss, "correct": abs(gap_pred - gap_gt).mean()}

    def validation_epoch_end(self, outputs):
        losses = [loss["loss"] for loss in outputs]
        correct = [loss["correct"] for loss in outputs]
        self.log("val_loss", torch.stack(losses).mean(), prog_bar=True)
        self.log("val_abs_diff", torch.stack(correct).mean(), prog_bar=True)

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
