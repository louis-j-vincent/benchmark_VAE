from typing import List

import numpy as np
import torch
import torch.nn as nn

from ..nn import BaseDecoder, BaseDiscriminator, BaseEncoder, BaseMetric
from ..base.base_utils import ModelOutput

class EncoderVAAE(BaseEncoder):
    """
    Encoder for Variance-Aware Autoencoder (VAAE). This encoder is designed to 
    transform input data into a latent space representation with a mean and variance, 
    while handling missing values.

    !! corresponds to Encoder_AE_w_variance on pdds platform code but with no Z_alpha !!

    Args:
        args: A configuration object containing the necessary parameters such as 
              input dimensions and latent dimension.
    """

    def __init__(self, args):
        super().__init__()
        self.input_dim = args.input_dim[1]
        self.latent_dim = args.latent_dim
        self.nan_value = -10.0  # Value to indicate missing data (NaNs)

        # Define the convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim * 10),
            nn.ReLU(),
            nn.Linear(self.input_dim * 10, 1024),
            nn.ReLU(),
        )

        # Define layers for embedding and log variance
        self.embedding = nn.Linear(1024, self.latent_dim)
        self.log_var = nn.Sequential(
            nn.Linear(1024, self.latent_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            output: A ModelOutput object containing the embedding and log covariance.
        """
        # Reshape input and create a mask for missing values
        x = x.reshape(x.shape[0], x.shape[-1])
        mask = (x != self.nan_value)

        # Concatenate input with its mask and pass through convolutional layers
        xU = torch.cat((x, mask), dim=1)
        h1 = self.conv_layers(xU).reshape(x.shape[0], -1)

        # Generate the embedding and log covariance
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output
