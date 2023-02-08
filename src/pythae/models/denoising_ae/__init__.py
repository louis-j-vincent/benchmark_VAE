"""Implementation of a Vanilla Autoencoder model.

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:

"""

from .denoising_ae_config import denoising_AEConfig
from .denoising_ae_model import denoising_AE

__all__ = ["denoising_AE","denoising_AEConfig"]
