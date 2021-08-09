#   ______                                           __                 
#  /      \                                         /  |                
# /$$$$$$  | __   __   __   ______   _______        $$ |       __    __ 
# $$ |  $$ |/  | /  | /  | /      \ /       \       $$ |      /  |  /  |
# $$ |  $$ |$$ | $$ | $$ |/$$$$$$  |$$$$$$$  |      $$ |      $$ |  $$ |
# $$ |  $$ |$$ | $$ | $$ |$$    $$ |$$ |  $$ |      $$ |      $$ |  $$ |
# $$ \__$$ |$$ \_$$ \_$$ |$$$$$$$$/ $$ |  $$ |      $$ |_____ $$ \__$$ |
# $$    $$/ $$   $$   $$/ $$       |$$ |  $$ |      $$       |$$    $$/ 
#  $$$$$$/   $$$$$/$$$$/   $$$$$$$/ $$/   $$/       $$$$$$$$/  $$$$$$/ 
#
# File: feature_encoding.py
# Author: Owen Lu
# Date: 2021/8/7
# Email: jiangxiluning@gmail.com
# Description:
import typing

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from fastai.vision.models.unet import DynamicUnet
import easydict

from .unet import UNetWithResnetEncoder


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class FeatureEncoding(nn.Module):

    def __init__(self, config: easydict.EasyDict):
        super(FeatureEncoding, self).__init__()
        self.lattice_dim = config.model.lattice_dim

        resnet = torchvision.models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(self.lattice_dim + 2,
                                 self.resnet.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.unet = UNetWithResnetEncoder(resnet)
        self.add_coords = AddCoords(with_r=False)

    def forward(self, lattice, b_t, *args, **kwargs):
        """

        Args:
            lattice: B C H W
            b_t: list of tensors [[N1, 2], [N2, 2], ..., [N_(B), 2]]
            *args:
            **kwargs:

        Returns:

        """

        x = self.add_coords(lattice)
        output = lattice + self.unet(x)
        output = output.permute((0, 2, 3, 1)).contiguous()
        F_features = []
        lengths = []
        for i, center in enumerate(b_t):
            # N_i, C
            f = output[i][center]
            lengths.append(f.shape[0])
            F_features.append(f)

        max_len = max(lengths)

        sequences = torch.zeros((len(F_features), max_len, output.shape[-1]), dtype=output.dtype).to(output.device)
        for i, feature in enumerate(F_features):
            sequences[i][:feature.shape[0]] = feature

        lengths = torch.tensor(lengths, dtype=torch.long, device=sequences.device)

        return sequences, lengths
