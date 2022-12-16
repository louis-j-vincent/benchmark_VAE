"""This module is the implementation of the Regularized AE with L2 decoder parameter regularization
proposed in (https://arxiv.org/abs/1903.12436).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .emae_config import EMAE_Config
from .emae_model import EMAE

__all__ = ["EMAE", "EMAE_Config"]
