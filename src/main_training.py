import hydra
from models.pl_wrapped.supervised_regression import BaselineSupervisedRegressor
from omegaconf import DictConfig
from data.s1t1module import ChemDataModule

from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks.early_stopping import EarlyStopping



@hydra.main(config_path="conf", config_name= "config")
def main(cfg : DictConfig):

    fold_idx = 0

    pl_model = BaselineSupervisedRegressor(cfg.trainer.opt, cfg.model, cfg.trainer.criterion, 1)
    datam = ChemDataModule(cfg.trainer, fold_idx = fold_idx)
    logger = pl_loggers.TensorBoardLogger(
            save_dir=cfg.trainer.log_dir
        )
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.trainer.checkpoint_dir + f"{fold_idx}",
        save_top_k=cfg.trainer.save_top_k,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
            # accelerator="ddp",
            # precision=16,
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





if __name__ == "__main__":
    main()