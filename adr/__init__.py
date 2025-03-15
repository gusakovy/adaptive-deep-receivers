"""
adaptive-deep-receivers
=======================

Online deep learning methods for adaptive receiver design.
"""

__version__ = "1.0.0"
__author__ = "Yakov Gusakov"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Package initialized")


from adr.src.detectors import DeepSICBlock, DeepSIC
from adr.src.trainers import minibatch_gradient_descent
from adr.src.channels import Channel, UplinkMimoChannel

__all__ = [
    DeepSICBlock,
    DeepSIC,

    minibatch_gradient_descent,

    Channel,
    UplinkMimoChannel
]
