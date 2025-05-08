import jax.numpy as jnp
from jax import Array

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