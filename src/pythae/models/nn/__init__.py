"""
In this module are stored the main Neural Networks Architectures.
"""


from .base_architectures import BaseDecoder, BaseDiscriminator, BaseEncoder, BaseMetric

from .new_architectures import EncoderVAAE

__all__ = ["BaseDecoder", "BaseEncoder", "BaseMetric", "BaseDiscriminator", "EncoderVAAE"]
