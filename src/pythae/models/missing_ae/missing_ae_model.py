import os
from typing import Optional

import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_AE_MLP
from .missing_ae_config import vAEConfig
from torch import tensor, cat, exp, std
import torch
from numpy.random import binomial
import numpy as np

class vAE(BaseAE):
    """
    Vanilla Autoencoder model with support for handling missing values through data corruption.

    Args:
        model_config (vAEConfig): Configuration object with model parameters.
        encoder (BaseEncoder): Optional encoder instance (inherits from `torch.nn.Module`).
        decoder (BaseDecoder): Optional decoder instance (inherits from `torch.nn.Module`).
    """

    def __init__(self, model_config: vAEConfig, encoder: Optional[BaseEncoder] = None, decoder: Optional[BaseDecoder] = None):
        super().__init__(model_config=model_config, decoder=decoder)

        self.model_name = "vAE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided! 'input_dim' must be set in model_config."
                )
            encoder = Encoder_AE_MLP(model_config)
            self.model_config.uses_default_encoder = True
        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

        # Parameters for data corruption and regularization
        self.nan_value = -10.0  # Value used to indicate NaNs in the data
        self.missing_value_prob = 0.1  # Probability for introducing missing values
        self.num_repeats = 10  # Number of repetitions for data augmentation
        self.initial_loss = True  # Flag for initial loss calculation
        self.regularization_weight = 1.0  # Weight for regularization loss
        self.update_parameters = None  # Placeholder for additional training parameters
        self.temperature = None  # Placeholder for temperature scaling

        # Check if the encoder returns a log covariance
        self.uses_log_covariance = hasattr(encoder, 'log_covariance')

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """
        Forward pass: encode, corrupt, decode, and compute losses.

        Args:
            inputs (BaseDataset): Dataset containing the input data.

        Returns:
            ModelOutput: Object containing loss, reconstruction, and latent variables.
        """
        x = inputs["data"]
        z = self.encoder(x).embedding

        # Repeat and corrupt data if training
        repeats = self.num_repeats if self.training else 3
        x_repeated, z_repeated_mu, z_repeated_sigma = self.repeat_and_corrupt_data(x, z, repeats)

        # Decode corrupted data
        recon_x = self.decoder(z_repeated_mu)["reconstruction"]
        recon_x = recon_x.squeeze(1) if recon_x.shape != x.shape else recon_x

        # Compute losses
        recon_loss = self.compute_reconstruction_loss(recon_x, x_repeated, x)
        reg_loss = self.loss_log_proba(z, z_repeated_mu, z_repeated_sigma)

        # Combine losses
        total_loss = recon_loss + self.regularization_weight * reg_loss

        return ModelOutput(loss=total_loss, recon_x=recon_x, z=z)

    def repeat_and_corrupt_data(self, x, z, repeats):
        """
        Repeat and corrupt the input data.

        Args:
            x (Tensor): Input data.
            z (Tensor): Latent representation of the input data.
            repeats (int): Number of times to repeat and corrupt the data.

        Returns:
            Tuple[Tensor, Tensor]: Corrupted data and repeated latent representation.
        """
        x_repeated = x.repeat_interleave(repeats, dim=0)
        z_repeated = z.repeat_interleave(repeats, dim=0)

        # Introduce missing values in the repeated data
        x_corrupted = self.corrupt_data(x_repeated)

        # Re-encode corrupted data
        z_reencoded = self.encoder(x_corrupted)
        z_repeated_mu = z_reencoded.embedding
        z_repeated_sigma = torch.exp(z_reencoded.log_covariance) if self.uses_log_covariance else None

        return x_repeated, z_repeated_mu, z_repeated_sigma

    def corrupt_data(self, x):
        """
        Introduce missing values in the input data with a given probability.

        Args:
            x (Tensor): Input data to corrupt.

        Returns:
            Tensor: Corrupted input data.
        """
        mask_existing = (x != self.nan_value)
        corruption_mask = torch.bernoulli(mask_existing * self.missing_value_prob)
        x_corrupted = x.clone()
        x_corrupted[corruption_mask == 1] = self.nan_value

        return x_corrupted

    def compute_reconstruction_loss(self, recon_x, x_repeated, x_original):
        """
        Compute the reconstruction loss using MSE.

        Args:
            recon_x (Tensor): Reconstructed data.
            x_repeated (Tensor): Repeated and corrupted input data.
            x_original (Tensor): Original input data.

        Returns:
            Tensor: Reconstruction loss.
        """
        mask_valid = (x_original != self.nan_value)
        return F.mse_loss(recon_x[mask_valid], x_repeated[mask_valid], reduction="mean")

    def loss_log_proba(self, z, z_mu, z_sigma=None):
        """
        Compute the log-probability-based loss for regularization.

        Args:
            z (Tensor): Original latent variables.
            z_mu (Tensor): Mean of re-encoded latent variables.
            z_sigma (Tensor, optional): Standard deviation of re-encoded latent variables.

        Returns:
            Tensor: Log-probability-based regularization loss.
        """
        if z_sigma is None:
            z_sigma = torch.ones_like(z_mu)  # Assume unit variance if not provided

        eps = 1e-5
        loss_mu = ((z - z_mu) ** 2 / (z_sigma + eps)).sum(dim=1).mean(dim=0)
        loss_sigma = torch.abs(z_sigma.mean() - 1)

        return loss_mu + loss_sigma
