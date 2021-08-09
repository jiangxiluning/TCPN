from typing import Optional
import pathlib

from easydict import EasyDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import wandb, tensorboard

from ..model.model import LitClassifier
from ..data_module.data_modules import MNISTDataModule
from .utils import get_valid_arguments


def test(ckpt_path: str,
         hparams_path: str,
         config: EasyDict,
         command_opts: Optional[dict] = None):

    data_module = MNISTDataModule(config)
    data_module.setup(stage='test')

    test_loader = data_module.test_dataloader()

    model = LitClassifier.load_from_checkpoint(checkpoint_path=ckpt_path,
                                               hparams_file=hparams_path,
                                               map_location=None)

    default_args = dict()
    additional_args = get_valid_arguments(pl.Trainer, command_opts)
    default_args.update(additional_args)
    default_args['logger'] = False
    default_args['callbacks'] = None
    trainer = pl.Trainer(**default_args)
    trainer.test(model, test_dataloaders=test_loader)
