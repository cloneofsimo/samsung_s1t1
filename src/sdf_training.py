import hydra
from models.pl_wrapped.supervised_regression_dim_2 import BaselineSupervisedRegressor
from omegaconf import DictConfig
from dataset.s1t1_sdf_module import SDFDataModule
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import pandas as pd


@hydra.main(config_path="conf", config_name="config_2")
def main(cfg: DictConfig):

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    for fold_idx in range(5):
        pl_model = BaselineSupervisedRegressor(
            cfg.trainer.opt, cfg.model, cfg.trainer.criterion, 2
        )
        datam = SDFDataModule(cfg.trainer, fold_idx=fold_idx)
        logger = pl_loggers.TensorBoardLogger(save_dir=cfg.trainer.log_dir)
        checkpoint_callback = callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=cfg.trainer.checkpoint_dir + f"{fold_idx}",
            save_top_k=cfg.trainer.save_top_k,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=40,
            verbose=True,
            mode="min",
        )

        trainer = Trainer(
            # accelerator="ddp",
            # precision=16,
            # resume_from_checkpoint="/home/simo/dl/comp2021/samsung_s1t1/src/outputs/2021-09-10/06-00-29/checkpoints0/epoch=129-step=6239.ckpt",
            max_epochs=cfg.trainer.epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            default_root_dir=cfg.trainer.log_dir,
            fast_dev_run=cfg.trainer.fast_dev_run,
            gpus=cfg.trainer.gpus,
            logger=logger,
            terminate_on_nan=True,
            weights_save_path=cfg.trainer.checkpoint_dir,
            check_val_every_n_epoch=cfg.trainer.check_val_freq,
            # val_check_interval=0.25,
        )
        trainer.fit(pl_model, datamodule=datam)


from tqdm import tqdm
from glob import glob
import numpy as np

CHEKCPOINT_PATHS = {
    "sch_1": "/home/simo/dl/comp2021/samsung_s1t1/src/outputs/2021-09-26/07-55-01/checkpoints*/*.ckpt",
    "sch_2": "/home/simo/dl/comp2021/samsung_s1t1/src/outputs/2021-09-26/15-48-16",
}


@hydra.main(config_path="conf", config_name="config")
def infer(cfg: DictConfig):
    fold_idx = 0

    pl_model = BaselineSupervisedRegressor(
        cfg.trainer.opt, cfg.model, cfg.trainer.criterion, 2
    )

    total_out = 0

    pbar = tqdm(glob())

    for ckpt in pbar:
        pbar.set_description(ckpt)

        state_dict = torch.load(ckpt)["state_dict"]
        new_state_dict = {}

        for k, v in state_dict.items():
            new_state_dict[k[6:]] = v

        pl_model.model.load_state_dict(new_state_dict)
        pl_model.model.eval()
        pl_model.model.to("cuda")

        datam = SDFDataModule(cfg.trainer, fold_idx=fold_idx)
        datam.setup("fit")

        ans = []
        for x in tqdm(datam.test_dataloader()):
            x = x.to("cuda")
            y = pl_model(x)
            gap = y["TARGET"]
            ans += list(gap.view(-1).detach().cpu().numpy())

        ans = torch.tensor(ans)
        # print(ans)
        total_out = total_out + ans

    submission = pd.read_csv(
        "/home/simo/dl/comp2021/samsung_s1t1/sample_submission.csv"
    )
    submission["ST1_GAP(eV)"] = (total_out.relu() / len(pbar)).tolist()
    submission.to_csv("submission.csv", index=False)


@hydra.main(config_path="conf", config_name="config")
def infer_embedding_train_test(cfg: DictConfig):
    fold_idx = 0

    pl_model = BaselineSupervisedRegressor(
        cfg.trainer.opt, cfg.model, cfg.trainer.criterion, 2
    )

    total_out = 0

    pbar = tqdm(
        glob(
            "/home/simo/dl/comp2021/samsung_s1t1/src/outputs/2021-09-26/07-55-01/checkpoints*/*.ckpt"
        )
    )

    for ckpt in pbar:
        pbar.set_description(ckpt)

        state_dict = torch.load(ckpt)["state_dict"]
        new_state_dict = {}

        for k, v in state_dict.items():
            new_state_dict[k[6:]] = v

        pl_model.model.load_state_dict(new_state_dict)
        pl_model.model.eval()
        pl_model.model.to("cuda")

        datam = SDFDataModule(cfg.trainer, fold_idx=fold_idx)
        datam.setup("fit")

        ans = []
        for x in tqdm(datam.test_dataloader()):
            x = x.to("cuda")
            y = pl_model(x)
            gap = y["TARGET"]
            ans += list(gap.view(-1).detach().cpu().numpy())

        ans = torch.tensor(ans)
        # print(ans)
        total_out = total_out + ans

    submission = pd.read_csv(
        "/home/simo/dl/comp2021/samsung_s1t1/sample_submission.csv"
    )
    submission["ST1_GAP(eV)"] = (total_out.relu() / len(pbar)).tolist()
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
    # infer()
