import hydra
from hydra.core.config_store import ConfigStore

from gen_names.config import Params, MambaModel, Scheduler_ReduceOnPlateau, Scheduler_OneCycleLR
from gen_names.models.mamba import Mamba_Lightning
from gen_names.data import NamesDataModule

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import os

cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="model", name="base_mamba", node=MambaModel)
cs.store(group="scheduler", name="base_rop", node=Scheduler_ReduceOnPlateau)
cs.store(group="scheduler", name="base_oclr", node=Scheduler_OneCycleLR)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    L.seed_everything(cfg.training.seed)
    wandb.login(key="dec2ee769ce2e455dd463be9b11767cf8190d658")
    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    dm = NamesDataModule(
        data_dir = cfg.data.data_directory,
        batch_size = cfg.training.batch,
        chunk_size = cfg.data.chunk_size
    )

    model = Mamba_Lightning(cfg)

    os.mkdir(working_dir + cfg.training.wandb_path)
    wandb_log = WandbLogger(
        project = cfg.training.project_name, 
        name = cfg.training.train_name, 
        save_dir = working_dir + cfg.training.wandb_path
    )

    checkpoint = ModelCheckpoint(
        dirpath = working_dir + cfg.training.model_path,
        filename = "epoch_{epoch}-{val_loss:.2f}-{val_bleu:.2f}",
        save_top_k = cfg.training.save_best_of,
        monitor = cfg.training.checkpoint_monitor
    )
    lr_monitor = LearningRateMonitor(logging_interval = "epoch")
    early_stop = EarlyStopping(monitor = cfg.training.checkpoint_monitor, patience = cfg.training.early_stopping_patience)

    trainer = L.Trainer(
        max_epochs = cfg.training.epochs,
        accelerator = "auto",
        log_every_n_steps = 10,
        devices = 1,
        logger = wandb_log,
        callbacks = [checkpoint, lr_monitor, early_stop],
        fast_dev_run = 5
    )
    trainer.fit(model = model, datamodule = dm)

    wandb.finish()


if __name__ == "__main__":
    main()