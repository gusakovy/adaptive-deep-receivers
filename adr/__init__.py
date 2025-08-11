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

from adr.src.detectors.deepsic_block import DeepSICBlock
from adr.src.detectors import DeepSIC, BayesianDeepSIC
from adr.src.detectors import ResNetDetector, BayesianResNetDetector
from adr.src.training_algorithms import build_sgd_train_fn, build_sgd_step_fn, iterative_ekf
from adr.src.training_algorithms import step_fn_builder 
from adr.src.channels import Channel, UplinkMimoChannel
from adr.src.utils import CovarianceType, Metric, TrainingMethod
from adr.src.utils import bit_array_to_index, index_to_bit_array, complex_to_stacked_real, stacked_real_to_complex

__all__ = [
    DeepSICBlock,
    DeepSIC,
    BayesianDeepSIC,
    ResNetDetector,
    BayesianResNetDetector,

    build_sgd_train_fn,
    build_sgd_step_fn,
    iterative_ekf,
    step_fn_builder,

    Channel,
    UplinkMimoChannel,

    CovarianceType,
    TrainingMethod,
    Metric,
    bit_array_to_index,
    index_to_bit_array,
    complex_to_stacked_real,
    stacked_real_to_complex,
]
