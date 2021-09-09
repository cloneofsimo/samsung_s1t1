import torch
import torch.nn as nn

import pytorch_lightning as pl
import numpy as np
from ..core.gine import GINE
from ..core.gat import GAT


criterions = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "sl1": nn.SmoothL1Loss(beta=0.002),
}


models = {"gine": GINE, "gat": GAT}

lr_sch = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "step": torch.optim.lr_scheduler.StepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}


class FineTuneRegressor(pl.LightningModule):
    def __init__(self, optconf, modelconf, criterionconf, n_ydim):
        super().__init__()
        self.model = models[modelconf.name](n_ydim=19, **modelconf)

        # Load model from pl checkpoint.
        checkpoint_path = "/home/simo/dl/comp2021/samsung_s1t1/src/outputs/2021-09-10/01-58-09/checkpoints0/epoch=299-step=30599.ckpt"
        state_dict = torch.load(checkpoint_path)["state_dict"]
        new_state_dict = {}

        for k, v in state_dict.items():
            new_state_dict[k[6:]] = v

        self.model.load_state_dict(new_state_dict)
        self.model.head = nn.Sequential(
            nn.Linear(self.model.head[0].in_features, 1024),
            nn.GELU(),
            # nn.Dropout(p=0.05),
            nn.Linear(1024, n_ydim),
        )

        self.model.main.requires_grad_ = False

        del state_dict, new_state_dict

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

    def validation_epoch_end(
        self,
        outputs,
    ):
        losses = [loss["loss"] for loss in outputs]
        correct = [loss["correct"] for loss in outputs]
        self.log("val_loss", torch.stack(losses).mean(), prog_bar=True)
        self.log("val_mae_loss", np.array(correct).mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.optconf.lr
        )

        return {
            "optimizer": optimizer,
        }
