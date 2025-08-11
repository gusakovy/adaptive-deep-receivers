"""
Detectors.
"""

from adr.src.detectors.deepsic import DeepSICBlock, DeepSIC
from adr.src.detectors.bayesian_deepsic import BayesianDeepSIC
from adr.src.detectors.resnet_detector import ResNetDetector

__all__ = [
    DeepSICBlock,
    DeepSIC,
    BayesianDeepSIC,
    ResNetDetector
]
