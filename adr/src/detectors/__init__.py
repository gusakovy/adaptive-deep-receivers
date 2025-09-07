"""
Detectors.
"""

from adr.src.detectors.base import Detector
from adr.src.detectors.deepsic import DeepSIC
from adr.src.detectors.resnet import ResNetDetector
__all__ = [
    Detector,
    DeepSIC,
    ResNetDetector,
]
