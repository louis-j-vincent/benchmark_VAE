"""
In this module are stored the main Neural Networks Architectures.
"""
import torch
import numpy as np
import torch.nn as nn
from typing import List

import torch.nn as nn
from torch import cat, stack

from ..base.base_utils import ModelOutput

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

class Encoder_vAE_distribution_shift(BaseEncoder): #corresponds to Encoder_AE_w_variance on pdds platform code but with no Z_alpha
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim[0]
        self.latent_dim = args.latent_dim
        self.ds_dim = 4

        self.conv_layers = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim*10),
            nn.ReLU(),
            nn.Linear(self.input_dim*10, 1024),
            nn.ReLU(),
        )

        self.distribution_shift = nn.Sequential(
            nn.Linear(1024 + self.ds_dim, 1024),
            nn.ReLU()
        )

        self.embedding = nn.Linear(1024, args.latent_dim)
        self.log_var = nn.Sequential(nn.Linear(1024, args.latent_dim),nn.ReLU())

    def forward(self, x: torch.Tensor, mfg: torch.Tensor):
        u = (x!=-10)
        xU = torch.cat((x,u),axis=1)
        h1 = self.conv_layers(xU).reshape(x.shape[0], -1)
        h1 = torch.cat((h1,mfg),axis=1)
        h2 = self.distribution_shift(h1)
        output = ModelOutput(
            embedding=self.embedding(h2),
            log_covariance=self.log_var(h2)
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

class Decoder_AE_distribution_shift(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim[0]
        self.latent_dim = args.latent_dim
        self.ds_dim = 4

        self.embedding_ = nn.Sequential(
            nn.Linear(self.latent_dim, 1024)
        )

        self.distribution_shift = nn.Sequential(
            nn.Linear(1024 + self.ds_dim, 1024),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.Linear(1024, self.input_dim*10),
            nn.ReLU(),
            nn.Linear(self.input_dim*10, self.input_dim),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor, mfg: torch.Tensor):
        z_ = self.embedding_(z)
        z_ = torch.cat((z_,mfg),axis=1)
        z_ = self.distribution_shift(z_)
        h1 = self.conv_layers(z_).reshape(z.shape[0], -1)
        output = ModelOutput(reconstruction=h1)

        return output

## quicker version (less layers)

class Encoder_vAE_distribution_shift(BaseEncoder): #corresponds to Encoder_AE_w_variance on pdds platform code but with no Z_alpha
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim[0]
        self.latent_dim = args.latent_dim
        self.ds_dim = 4

        self.conv_layers = nn.Sequential(
            nn.Linear(self.input_dim * 2 + self.ds_dim, self.input_dim*10),
            nn.ReLU(),
            nn.Linear(self.input_dim*10, 1024),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(1024, args.latent_dim)
        self.log_var = nn.Sequential(nn.Linear(1024, args.latent_dim),nn.ReLU())

    def forward(self, x: torch.Tensor, mfg: torch.Tensor):
        u = (x!=-10)
        xU = torch.cat((x,u,mfg),axis=1)
        h1 = self.conv_layers(xU).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output

class Decoder_AE_distribution_shift(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.ds_dim = 4

        # assert 0, np.prod(args.input_dim)
        inter_layer = 64

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(nn.Linear(args.latent_dim + self.ds_dim, inter_layer), nn.ReLU()),
            )
        #layers.append(nn.Sequential(nn.Linear(inter_layer, inter_layer), nn.ReLU()))


        layers.append(
            nn.Sequential(nn.Linear(inter_layer, int(np.prod(args.input_dim))))
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, mfg: torch.Tensor, output_layer_levels: List[int] = None):

        z = torch.cat((z,mfg),axis=1)

        output = ModelOutput()

        max_depth = self.depth

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output

