"""
In this module are stored the main Neural Networks Architectures.
"""


import torch
import torch.nn as nn

from ..base.base_utils import ModelOutput

from .base_architectures import BaseDecoder, BaseDiscriminator, BaseEncoder, BaseMetric

import torch
import numpy as np
import torch.nn as nn
from typing import List

import torch.nn as nn
from torch import cat, stack

class Encoder_VAE_Z_alpha(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim[1]
        self.latent_dim = args.latent_dim

        self.conv_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*10),
            nn.ReLU(),
            nn.Linear(self.input_dim*10, 1024),
            nn.ReLU(),
        )

        self.mu = nn.Linear(1024, args.latent_dim)

    def forward(self, x: torch.Tensor):

        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.mu(h1)
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

        self.encoder_alpha = Encoder_VAE_Z_alpha(args)

    def forward(self, x: torch.Tensor):
        u = (x!=-10)
        xU = cat((x,u),axis=1)
        h1 = self.conv_layers(xU).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output

class Decoder_AE_missing(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim[0]
        self.latent_dim = args.latent_dim

        self.fc = nn.Linear(args.latent_dim, 1024)
        self.deconv_layers = nn.Sequential(
            nn.Linear(1024, self.input_dim*10),
            nn.ReLU(),
            nn.Linear(self.input_dim*10, self.input_dim),
        )

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], 1024)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))
        
__all__ = ["BaseDecoder", "BaseEncoder", "BaseMetric", "BaseDiscriminator",
          "Encoder_VAE_Z_alpha","Encoder_VAE_missing","Decoder_AE_missing"]

        return output
