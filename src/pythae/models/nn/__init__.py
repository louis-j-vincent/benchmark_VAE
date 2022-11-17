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

class Encoder_vAE_MNIST(BaseEncoder):
    """
    A proposed Convolutional encoder Neural net suited for MNIST and Autoencoder-based models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.mnist import Encoder_AE_MNIST
            >>> from pythae.models import AEConfig
            >>> model_config = AEConfig(input_dim=(1, 28, 28), latent_dim=16)
            >>> encoder = Encoder_AE_MNIST(model_config)
            >>> encoder
            ... Encoder_AE_MNIST(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=1024, out_features=16, bias=True)
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import AE
        >>> model = AE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True

    .. note::

        Please note that this encoder is only suitable for Autoencoder based models since it only
        outputs the embeddings of the input data under the key `embedding`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 1, 28, 28)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 16])


    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 2

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1), nn.BatchNorm2d(512), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1), nn.BatchNorm2d(1024), nn.ReLU()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(1024, args.latent_dim)
        self.log_var = nn.Sequential(nn.Linear(1024, args.latent_dim),nn.ReLU())


    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        u = (x!=-10)
        xU = cat((x,u),axis=1)
        out = xU

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))

        return output


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
        xU = cat((x,u),axis=1)
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

class Encoder_VAE_Z_alpha(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim[0]
        self.input_dim_alpha = args.input_dim[1]
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

        self.encoder_alpha = nn.Sequential(
            nn.Linear(self.input_dim_alpha, self.input_dim*10),
            nn.ReLU(),
            nn.Linear(self.input_dim*10, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.latent_dim)
        )
        self.merging_net = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim*10),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 10, self.latent_dim)
        )

    def forward(self, x: torch.Tensor):
        u = (x!=-10)
        xU = cat((x,u),axis=1)
        h1 = self.conv_layers(xU).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output

class Decoder_AE(BaseDecoder):
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
            
__all__ = ["BaseDecoder", "BaseEncoder", "BaseMetric", "BaseDiscriminator",
          "Encoder_VAE_Z_alpha","Encoder_VAE_missing","Encoder_VAE_Z_alpha","Decoder_AE",'Encoder_vAE']
