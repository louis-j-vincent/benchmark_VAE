"""
In this module are stored the main Neural Networks Architectures.
"""


import torch
import torch.nn as nn

from ..base.base_utils import ModelOutput


class BaseEncoder(nn.Module):
    """This is a base class for Encoders neural networks.
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of 
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own encoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseEncoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Encoder(BaseEncoder):
            ...
            ...     def __init__(self):
            ...         BaseEncoder.__init__(self)
            ...         # your code
            ...
            ...     def forward(self, x: torch.Tensor):
            ...         # your code
            ...         output = ModelOutput(
            ...             embedding=embedding,
            ...             log_covariance=log_var # for VAE based models
            ...         )
            ...         return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()


class BaseDecoder(nn.Module):
    """This is a base class for Decoders neural networks.
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, z: torch.Tensor):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of 
         :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own decoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseDecoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_decoder(BaseDecoder):
            ...
            ...    def __init__(self):
            ...        BaseDecoder.__init__(self)
            ...        # your code
            ...
            ...    def forward(self, z: torch.Tensor):
            ...        # your code
            ...        output = ModelOutput(
            ...             reconstruction=reconstruction
            ...         )
            ...        return output

        Parameters:
            z (torch.Tensor): The latent data that must be decoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the decoder
           
        .. note::

            By convention, the reconstruction tensors should be in [0, 1] and of shape 
            BATCH x channels x ...

        """
        raise NotImplementedError()

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

        return output
