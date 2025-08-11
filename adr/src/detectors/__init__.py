"""
Detectors.
"""

from adr.src.detectors.deepsic import DeepSICBlock, DeepSIC
from adr.src.detectors.bayesian_deepsic import BayesianDeepSIC
from adr.src.detectors.resnet_detector import ResNetDetector
from adr.src.detectors.bayesian_resnet_detector import BayesianResNetDetector

__all__ = [
    DeepSICBlock,
    DeepSIC,
    BayesianDeepSIC,
    ResNetDetector,
    BayesianResNetDetector
]
