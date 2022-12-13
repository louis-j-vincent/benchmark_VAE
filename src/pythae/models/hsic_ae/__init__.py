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

from .ic_ae_config import IC_AEConfig
from .ic_ae_model import IC_AE

__all__ = ["IC_AE","IC_AEConfig"]
