from enum import Enum
from collections import deque
import jax.numpy as jnp
from jax import Array


class CovarianceType(Enum):
    FULL = "full"
    DG = "diagonal"

class TrainingMethod(Enum):
    """Enumeration of training methods."""
    GD = "gd"      # (Streaming) Gradient Descent
    SGD = "sgd"    # (Batch) Stochastic Gradient Descent
    BBB = "bbb"    # Bayes by Backprop
    BLR = "blr"    # Bayesian Learning Rule
    BOG = "bog"    # Bayesian Online Gradient
    BONG = "bong"  # Bayesian Online Natural Gradient

class Metric:
    def __init__(self, history=100):
        self.history = history
        self.reset()

    def reset(self):
        self.values = deque(maxlen=self.history)

    def update(self, val):
        self.values.append(val)

    def get_stat(self, stat: str):
        if len(self.values) == 0:
            return 0
        values_list = list(self.values)
        if stat == "avg" or stat == "mean":
            return sum(values_list) / len(values_list)
        elif stat == "max":
            return max(values_list)
        elif stat == "min":
            return min(values_list)
        elif stat == "p50":
            return sorted(values_list)[len(values_list) // 2]
        elif stat == "p99":
            index = int(len(values_list) * 0.99) - 1
            return sorted(values_list)[max(0, index)]
        else:
            raise ValueError(f"Unsupported statistic: {stat}")

def bit_array_to_index(bit_vector: Array) -> Array:
    """
    Convert a vector of bits (in the last dimension) to an integer index.

    Args:
        bit_vector (Array): An array where the last dimension represents bits.

    Returns:
        Array: A JAX array of integer indices.
    """
    powers_of_two = 2 ** jnp.arange(bit_vector.shape[-1] - 1, -1, -1)
    return jnp.sum(bit_vector * powers_of_two, axis=-1)

def index_to_bit_array(index: Array, num_bits: int) -> Array:
    """
    Convert an integer index to a vector of bits.

    Args:
        index (Array): A JAX array of integer indices.
        num_bits (int): The number of bits in the output array.

    Returns:
        Array: An array where the last dimension represents bits.
    """
    bit_array = jnp.zeros((index.shape[0], num_bits), dtype=jnp.int32)
    for i in range(num_bits):
        bit_array = bit_array.at[:, i].set((index // (2 ** (num_bits - 1 - i))) % 2)
    return bit_array

def complex_to_stacked_real(complex_array: Array) -> Array:
    """
    Convert a complex array to a stacked real array.

    Args:
        complex_array (Array): An array of complex numbers.

    Returns:
        Array: An array where the real and imaginary parts are stacked along the last dimension.
    """
    return jnp.concatenate((complex_array.real, complex_array.imag), axis=-1)

def stacked_real_to_complex(stacked_array: Array) -> Array:
    """
    Convert a stacked real array to a complex array.

    Args:
        stacked_array (Array): An array where the real and imaginary parts are stacked along the last dimension.

    Returns:
        Array: An array of complex numbers.
    """
    if stacked_array.shape[-1] % 2 != 0:
        raise ValueError("The last dimension of the input array must be even.")
    mid = stacked_array.shape[-1] // 2
    return stacked_array[..., :mid] + 1j * stacked_array[..., mid:]