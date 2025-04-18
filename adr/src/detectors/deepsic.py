import os
from enum import Enum
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from flax import linen as nn
from adr.src.detectors.base import Detector


class CovarianceType(Enum):
    NONE = None
    FULL = "full"
    DG = "diagonal"


class DeepSICBlock(nn.Module):
    """Single block of a DeepSIC model.
    
    Args:
        symbol_bits (int): Number of bits per symbol.
        num_users (int): Number of users.
        num_antennas (int): Number of receive antennas.
        hidden_dim (int): Size of the hidden layer of the block.
        activation (callable, optional): Activation function. Defaults to ReLU.
    """

    symbol_bits: int
    num_users: int
    num_antennas: int
    hidden_dim: int
    activation: callable = nn.relu

    def setup(self):
        self.features = [self.hidden_dim, self.symbol_bits]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        return nn.Dense(self.features[-1])(x)


class DeepSIC(Detector):
    """DeepSIC model from https://ieeexplore.ieee.org/document/9242305 with bitwise outputs and 
    connections between blocks of the same user in adjacent layers.

    Args:
        key (int): Random key for parameter initialization.
        symbol_bits (int): Number of bits per symbol.
        num_users (int): Number of users.
        num_antennas (int): Number of receive antennas.
        num_layers (int): Number of soft interference cancellation (SIC) layers.
        hidden_dim (int): Size of the hidden layer of each block.
        cov_type (CovarianceType, optional): Type of covariance for the parameters. Defaults to CovarianceType.NONE.
    """

    def __init__(
            self,
            key: int | Array,
            symbol_bits: int,
            num_users: int,
            num_antennas: int,
            num_layers: int,
            hidden_dim: int,
            cov_type: CovarianceType = CovarianceType.NONE,
        ):
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.num_layers = num_layers
        self.rx_size = num_antennas if self.symbol_bits == 1 else 2 * self.num_antennas
        self.block_input_size = self.rx_size + self.symbol_bits * self.num_users
        self.block_model= DeepSICBlock(
            symbol_bits=self.symbol_bits,
            num_users=self.num_users,
            num_antennas=self.num_antennas,
            hidden_dim=hidden_dim
        )
        self.unravel_fn = None
        key = jr.PRNGKey(key) if isinstance(key, int) else key
        subkeys = jr.split(key, self.num_users * self.num_layers)
        flat_params_list = []
        for layer_idx in range(self.num_layers):
            for user_idx in range(self.num_users):
                subkey = subkeys[layer_idx * self.num_users + user_idx]
                params = self.block_model.init(subkey, jnp.empty((1, self.block_input_size)))
                flat_params, self.unravel_fn = ravel_pytree(params)
                flat_params_list.append(flat_params)

        self.params = jnp.stack(flat_params_list).reshape((self.num_layers, self.num_users, -1))
        self.param_size = self.params.shape[-1]
        self.cov_type = cov_type
        self.params_cov = None

    def init_params_cov(self, init_cov: float | Array):
        """Initialize the covariance matrices for the parameters of each DeepSICBlock.

        Args:
            init_cov (float | Array): Initial covariance scale or matrix or array of matrices of size num_layers x num_users.
        """
        cov_shape = (self.param_size, self.param_size) if self.cov_type == CovarianceType.FULL else (self.param_size,)
        if isinstance(init_cov, float) or init_cov.shape == cov_shape:
            if isinstance(init_cov, float):
                init_cov = init_cov * jnp.eye(self.param_size) if self.cov_type == CovarianceType.FULL else init_cov * jnp.ones(self.param_size)
            self.params_cov = jnp.tile(init_cov, (self.num_layers, self.num_users, 1, 1))
            self.params_cov = self.params_cov.reshape(self.num_layers, self.num_users, *cov_shape)
        elif init_cov.shape == (self.num_layers, self.num_users, *cov_shape):
            self.params_cov = init_cov
        else:
            raise ValueError(f"Invalid shape for init_cov. Expected float, or array of shape {cov_shape}"
                             f"or (num_layers, num_users, ...) = ({self.num_layers},{self.num_users},{cov_shape[0]},{cov_shape[1]}) "
                             f"but got {init_cov.shape} instead.")

    def _pred_and_rx_to_input(self, layer_num: int, rx: Array, pred: Array = None) -> Array:
        """Prepare shared input(s) for all blocks in a layer.

        Args:
            layer_num (int): Layer number.
            rx (Array): Received signal(s).
            pred (Array, optional): Soft decisions of the previous layer. Defaults to None for the first layer.

        Returns:
            Array: Shared input(s) for all blocks in a layer.
        """
        if layer_num == 0:
            initial_pred = jnp.full((1, self.symbol_bits * self.num_users), 0.5)
            pred = jnp.tile(initial_pred, (rx.shape[0], 1))

        else:
            pred = pred.transpose((1, 0, 2)).reshape((pred.shape[1], -1))

        return jnp.concatenate([pred, rx], axis=-1)

    def _block_soft_decode(self, params: Array, inputs: Array) -> Array:
        """Soft-decode using a single block.

        Args:
            params (Array): Block parameters.
            inputs (Array): Block input(s).

        Returns:
            Array: Soft decisions of the block.
        """
        unraveled_params = self.unravel_fn(params)
        bitwise_logits = self.block_model.apply(unraveled_params, inputs)
        return jax.nn.sigmoid(bitwise_logits)

    def layer_transition(self, layer_num: int, rx: Array, pred: Array = None) -> Array:
        """Pass data through a layer.

        Args:
            layer_num (int): Layer number.
            rx (Array): Received signal(s).
            pred (Array, optional): Soft decisions from the previous layer. Defaults to None.

        Returns:
            Array: Per symbol-bit soft decisions for all blocks in the layer.
        """
        inputs = self._pred_and_rx_to_input(layer_num, rx, pred)
        layer_outputs = jax.vmap(self._block_soft_decode, in_axes=(0, None))(self.params[layer_num], inputs)

        return jax.lax.stop_gradient(layer_outputs)

    def soft_decode(self, rx: Array, first_layers: int = None) -> Array:
        """Soft-decode a (batch of) received signal(s).

        Args:
            rx (Array): Received signal(s).
            first_layers (int, optional): Number of layers to use for decoding. Defaults to None to use all of them.

        Returns:
            Array: Per user and per symbol-bit soft decisions.
        """
        rx = jnp.atleast_2d(rx)
        predictions = None
        num_layers = self.num_layers if first_layers is None else min(first_layers, self.num_layers)
        for layer_idx in range(num_layers):
            predictions = self.layer_transition(layer_idx, rx, predictions)
        return predictions.transpose((1, 0, 2))

    def fit(self, rx: Array, labels: Array, train_block_fn: callable, **kwargs) -> None:
        """Train each block independently using the provided train_block_fn method.
        Training is performed layer-by-layer, with parallel updates for user blocks within each layer.

        Args:
            rx (Array): Received signal(s).
            labels (Array): True labels corresponding to the received signal(s).
            train_block_fn (callable): Function to train a single block.
            **kwargs: Additional arguments for the training function.
        """
        rx = jnp.atleast_2d(rx)
        labels = labels.reshape((rx.shape[0], self.num_users, self.symbol_bits))

        if self.cov_type != CovarianceType.NONE and self.params_cov is None:
            self.init_params_cov(0.0)

        def update_user_block(user_params, inputs, labels):
            return train_block_fn(
                user_params,
                self.unravel_fn,
                self.block_model.apply,
                inputs,
                labels,
                **kwargs
            )

        def train_layer(layer_idx, pred):
            layer_params = self.params[layer_idx]
            inputs = self._pred_and_rx_to_input(layer_idx, rx, pred)
            new_layer_params = jax.vmap(update_user_block, in_axes=(0, None, 1))(layer_params,inputs,labels)

            if self.cov_type != CovarianceType.NONE:
                self.params = self.params.at[layer_idx].set(new_layer_params[0])
                self.params_cov = self.params_cov.at[layer_idx].set(new_layer_params[1])
            else:
                self.params = self.params.at[layer_idx].set(new_layer_params)

            new_pred = self.layer_transition(layer_idx, rx, pred)
            return new_pred

        # Loop over the layers and train each layer.
        predictions = None
        for layer_idx in range(self.num_layers):
            predictions = train_layer(layer_idx, predictions)

    def save(self, path: str):
        """Save the detector parameters to disk.

        Args:
            path (str): Save path for the detector parameters.
        """
        jnp.save(path, self.params)

    def load(self, path: str):
        """Load the detector parameters from disk.

        Args:
            path (str): Path to detector parameters file.
        """
        self.params = jnp.load(path)
