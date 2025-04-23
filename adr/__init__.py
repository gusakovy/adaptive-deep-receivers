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

from adr.src.detectors import CovarianceType,DeepSICBlock, DeepSIC
from adr.src.training_algorithms import minibatch_sgd, streaming_gd, iterative_ekf, fg_bong, dg_bong, fg_bog, dg_bog, fg_bbb, dg_bbb
from adr.src.channels import Channel, UplinkMimoChannel

__all__ = [
    CovarianceType,
    DeepSICBlock,
    DeepSIC,

    minibatch_sgd,
    streaming_gd,
    iterative_ekf,
    fg_bong,
    dg_bong,
    fg_bog,
    dg_bog,
    fg_bbb,
    dg_bbb,

    Channel,
    UplinkMimoChannel
]
