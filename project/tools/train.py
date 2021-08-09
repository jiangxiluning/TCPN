from typing import Optional
import pathlib

from easydict import EasyDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import wandb, tensorboard

from ..model.model import LitClassifier
from ..data_module.data_modules import MNISTDataModule
from .utils import get_valid_arguments


def train(config: EasyDict,
          command_opts: Optional[dict] = None):
    pl.seed_everything(config.trainer.seed)

    data_module = MNISTDataModule(config)
    data_module.setup(stage='train')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = LitClassifier(config)

    # ------------
    # training
    # ------------

    default_args = dict(default_root_dir=pathlib.Path(config.trainer.output_dir).absolute(),
                        gpus=config.trainer.gpus,
                        log_gpu_memory=config.trainer.log_gpu_memory,
                        val_check_interval=config.trainer.val_check_interval,
                        num_nodes=config.trainer.num_nodes,
                        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
                        max_epochs=config.trainer.max_epochs,
                        log_every_n_steps=config.trainer.log_every_n_steps,
                        precision=config.trainer.precision,
                        flush_logs_every_n_steps=config.trainer.flush_logs_every_n_steps,
                        benchmark=config.trainer.benchmark,
                        deterministic=config.trainer.deterministic,
                        accelerator=config.trainer.accelerator
                        )

    additional_args = get_valid_arguments(pl.Trainer, command_opts)
    default_args.update(additional_args)

    loggers = []
    if config.trainer.loggers.wandb:
        wandb_logger = wandb.WandbLogger(project=config.system.model_name,
                                         name=config.system.run_name,
                                         offline=True,
                                         save_dir=pathlib.Path(config.trainer.output_dir) / 'wandb',
                                         config=config)

        loggers.append(wandb_logger)

    if config.trainer.loggers.tensorboard:
        tfb_logger = tensorboard.TensorBoardLogger(save_dir=pathlib.Path(config.trainer.output_dir),
                                                   name=config.system.model_name,
                                                   version=config.system.run_name)
        loggers.append(tfb_logger)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(filename=config.system.model_name + '_' + config.system.run_name,
                                                 period=config.trainer.checkpoints.period,
                                                 dirpath=pathlib.Path(config.trainer.output_dir)
                                                         / config.system.model_name
                                                         / config.system.run_name)

    default_args['logger'] = loggers
    default_args['callbacks'] = [checkpoint_cb]
    trainer = pl.Trainer(**default_args)
    trainer.fit(model, train_loader, val_loader)
