"""
Trainers.
"""

from adr.src.training_algorithms.sgd import minibatch_sgd, streaming_gd
from adr.src.training_algorithms.ekf import iterative_ekf
from adr.src.training_algorithms.bong import fg_bong, dg_bong
from adr.src.training_algorithms.bog import fg_bog, dg_bog
from adr.src.training_algorithms.bbb import fg_bbb, dg_bbb

__all__ = [
    minibatch_sgd,
    streaming_gd,
    iterative_ekf,
    fg_bong,
    dg_bong,
    fg_bog,
    dg_bog,
    fg_bbb,
    dg_bbb
]
