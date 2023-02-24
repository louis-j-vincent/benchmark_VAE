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

from .vfemae_config import vfEMAEConfig
from .vfemae_model import vfEMAE

__all__ = ["vfEMAE","vfEMAEConfig"]
