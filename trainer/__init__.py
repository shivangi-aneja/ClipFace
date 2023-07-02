import os
import torch
import wandb
import datetime
from pathlib import Path
from random import randint
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from util.filesystem_logger import FilesystemLogger


def generate_experiment_name(name, config):
    if config.resume is not None:
        experiment = Path(config.resume).parents[1].name
        os.environ['experiment'] = experiment
    elif not os.environ.get('experiment'):
        experiment = f"{datetime.datetime.now().strftime('%d%m%H%M')}_{name}_{config.experiment}"
        os.environ['experiment'] = experiment
    else:
        experiment = os.environ['experiment']
    return experiment


def create_trainer(name, config):
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = generate_experiment_name(name, config)
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed)

    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)
    logger = WandbLogger(project=f'{name}{config.suffix}', name=config.experiment, id=config.experiment, settings=wandb.Settings(start_method='thread'))
    checkpoint_callback = ModelCheckpoint(dirpath=(config.base_dir / Path("runs") / config.experiment / "checkpoints"), filename='_{epoch}', save_last=True, save_top_k=-1, verbose=False, every_n_epochs=config.save_epoch)
    gpu_count = torch.cuda.device_count()

    if gpu_count > 1:
        trainer = Trainer(devices=-1,
                          strategy='ddp',
                          accelerator='gpu',
                          log_every_n_steps=config.log_every_n_steps,
                          num_sanity_val_steps=config.sanity_steps,
                          max_epochs=config.max_epoch,
                          limit_val_batches=config.val_check_percent,
                          callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)),
                          check_val_every_n_epoch=max(1, config.val_check_interval),
                          resume_from_checkpoint=config.resume,
                          logger=logger,
                          benchmark=True)
    else:
        trainer = Trainer(devices=[0],
                          accelerator='gpu',
                          num_sanity_val_steps=config.sanity_steps,
                          log_every_n_steps=config.log_every_n_steps,
                          max_epochs=config.max_epoch,
                          limit_val_batches=config.val_check_percent,
                          callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)),
                          check_val_every_n_epoch=max(1, config.val_check_interval),
                          resume_from_checkpoint=config.resume,
                          logger=logger,
                          benchmark=True)
    return trainer
