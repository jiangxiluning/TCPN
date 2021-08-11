from typing import List, Optional, Any
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.nn as nn
from easydict import EasyDict

import torchmetrics
from .feature_encoding import FeatureEncoding

class TCPN(pl.LightningModule):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.save_hyperparameters()
        self.feature_encoder = FeatureEncoding(config)

        self.mode = config.model.mode
        self.num_classes = 4
        self.lattice_dim = config.model.lattice_dim
        self.hidden_dim = config.model.hidden_dim
        self.max_len = config.model.max_len

        self.tag_projection = nn.Linear(self.lattice_dim, self.num_classes)

    def forward(self, batch, batch_idx):
        lattice = batch[0]
        class_embed = batch[1]
        b_t = batch[2]

        sequences, lengths = self.feature_encoder(lattice, b_t)


        cp_output = None
        tag_output = None
        if self.training:
            pass
        else:

            if self.mode == 0:  # CP Mode
                pass
            elif self.mode == 1:  # tag mode
                pass
            else:
                raise NotImplementedError('Mode {} is not supported.'.format(self.mode))

        return dict(cp_output=cp_output, tag_output=tag_output)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)
        y_hat = torch.softmax(y_hat, dim=-1)
        self.acc(y_hat, y)

    def validation_epoch_end(self, *args) -> None:
        acc = self.acc.compute()
        self.log('val_acc', acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=-1)
        self.acc(y_hat, y)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        acc = self.acc.compute()
        self.log('test_acc', acc)

    def configure_optimizers(self):
        if self.hparams.config.trainer.optim.name == 'Adam':
            return torch.optim.Adam(self.parameters(), **self.hparams.config.trainer.optim.args)