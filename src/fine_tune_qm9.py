import hydra
from models.pl_wrapped.fine_tune_QM9_Gine import FineTuneRegressor
from omegaconf import DictConfig
from dataset.s1t1module import ChemDataModule

from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import pandas as pd


class FeatureExtractorFreezeUnfreeze(callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.model.main)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            for g in optimizer.param_groups:
                g["lr"] = 0.0001
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.main,
                optimizer=optimizer,
                train_bn=True,
            )


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    fold_idx = 0

    pl_model = FineTuneRegressor(cfg.trainer.opt, cfg.model, cfg.trainer.criterion, 1)
    datam = ChemDataModule(cfg.trainer, fold_idx=fold_idx)
    logger = pl_loggers.TensorBoardLogger(save_dir=cfg.trainer.log_dir)
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.trainer.checkpoint_dir + f"{fold_idx}",
        save_top_k=cfg.trainer.save_top_k,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode="min",
    )

    finetune_callback = FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=10)

    trainer = Trainer(
        # accelerator="ddp",
        # precision=16,
        resume_from_checkpoint="/home/simo/dl/comp2021/samsung_s1t1/src/outputs/2021-09-10/05-16-37/checkpoints0/epoch=161-step=15389.ckpt",
        max_epochs=cfg.trainer.epochs,
        callbacks=[checkpoint_callback, finetune_callback],
        # callbacks=[checkpoint_callback, finetune_callback],
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


@hydra.main(config_path="conf", config_name="config")
def infer(cfg: DictConfig):
    fold_idx = 0

    pl_model = BaselineSupervisedRegressor(
        cfg.trainer.opt, cfg.model, cfg.trainer.criterion, 1
    )

    checkpoint_path = "/home/simo/dl/comp2021/samsung_s1t1/src/outputs/2021-09-08/21-31-44/checkpoints0/epoch=131-step=25079.ckpt"
    state_dict = torch.load(checkpoint_path)["state_dict"]
    new_state_dict = {}

    for k, v in state_dict.items():
        new_state_dict[k[6:]] = v

    pl_model.model.load_state_dict(new_state_dict)
    pl_model.model.eval()
    pl_model.model.to("cuda")

    datam = ChemDataModule(cfg.trainer, fold_idx=fold_idx)
    datam.setup("fit")

    ans = []
    for x in tqdm(datam.test_dataloader()):
        x = x.to("cuda")
        y = pl_model(x)
        ans += list(y.view(-1).detach().cpu().numpy())

    submission = pd.read_csv(
        "/home/simo/dl/comp2021/samsung_s1t1/sample_submission.csv"
    )
    submission["ST1_GAP(eV)"] = ans
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
    # infer()
