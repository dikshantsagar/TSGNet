import sys
sys.path.append("../")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from os import getcwd, makedirs, environ
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from trainers.TSGNet_trainer import TSGNetTrainer
from datamodules.nonabberated_datamodule import NonAbberatedDataModule


@rank_zero_only
def update_config(logger, config_dict):
    logger.experiment.config.update(config_dict, allow_val_change=True)

def main(learning_rate, weight_decay, batch_size, gpus, max_epochs, downsample_rate, name):

    wandb_logger = WandbLogger(project='TSGNet', name=name)
    update_config(wandb_logger, {   "Dataset": "Non-Abberated",
                                    'lr' : learning_rate,
                                    'weight_decay': weight_decay,
                                    "batch_size" : batch_size,
                                    'num_gpus': gpus,
                                    "max_epochs": max_epochs,
                                    "downsample_rate": downsample_rate
    })

    master = "NODE_RANK" not in environ
    distributed_backend = DDPStrategy(
        find_unused_parameters=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(dirpath=f'../runs/non_abberated/{name}/', monitor='Loss/test')

    trainer = pl.Trainer(
                max_epochs=max_epochs,
                strategy=distributed_backend,
                devices=gpus if gpus > 0 else None,
                val_check_interval=1.0,
                log_every_n_steps=3,
                logger=wandb_logger,
                callbacks=[lr_monitor, model_checkpoint],
                
    )

    model = TSGNetTrainer(lr=learning_rate, weight_decay=weight_decay, max_epochs=max_epochs)
    dm = NonAbberatedDataModule(directory="/baldig/physicsprojects/electron_microscopy", downsample_rate=downsample_rate, batch_size=batch_size)

    trainer.fit(model, datamodule=dm)

    wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--downsample_rate", type=int, default=1)
    parser.add_argument("--name", type=str, help='Experiment Name')
    
    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    main(**parser.parse_args().__dict__)
