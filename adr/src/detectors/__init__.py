"""
Detectors.
"""

from adr.src.detectors.deepsic import DeepSICBlock, DeepSIC
from adr.src.detectors.bayesian_deepsic import BayesianDeepSIC

__all__ = [
    DeepSICBlock,
    DeepSIC,
    BayesianDeepSIC
]
