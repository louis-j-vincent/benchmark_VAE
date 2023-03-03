"""
In this module are stored the main Neural Networks Architectures.
"""
import torch
import numpy as np
import torch.nn as nn
from typing import List

import torch.nn as nn
from torch import cat, stack

from .base_architectures import BaseDecoder, BaseDiscriminator, BaseEncoder, BaseMetric

__all__ = ["BaseDecoder", "BaseEncoder", "BaseMetric", "BaseDiscriminator"]

class Encoder_vAE(BaseEncoder): #corresponds to Encoder_AE_w_variance on pdds platform code but with no Z_alpha
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim[0]
        self.latent_dim = args.latent_dim

        self.conv_layers = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim*10),
            nn.ReLU(),
            nn.Linear(self.input_dim*10, 1024),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(1024, args.latent_dim)
        self.log_var = nn.Sequential(nn.Linear(1024, args.latent_dim),nn.ReLU())

    def forward(self, x: torch.Tensor):
        u = (x!=-10)
        xU = torch.cat((x,u),axis=1)
        h1 = self.conv_layers(xU).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output

class Encoder_VAE_missing(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim[0]
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.conv_layers = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim*10),
            nn.ReLU(),
            nn.Linear(self.input_dim*10, 1024),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(1024, args.latent_dim)
        self.log_var = nn.Linear(1024, args.latent_dim)

    def forward(self, x: torch.Tensor):
        u = (x!=-10)
        xU = cat((x,u),axis=1)
        h1 = self.conv_layers(xU).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output

