import pickle
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from flax import linen as nn
from adr.src.detectors.base import Detector


class ResNetBlock(nn.Module):
    """ResNet block with skip connections."""
    hidden_dim: int
    activation: callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x

        # First dense layer
        x = nn.Dense(self.hidden_dim)(x)
        x = self.activation(x)

        # Second dense layer
        x = nn.Dense(self.hidden_dim)(x)

        # Skip connection (if dimensions match)
        if x.shape[-1] == residual.shape[-1]:
            x = x + residual

        x = self.activation(x)
        return x


class ResNetDetectorModel(nn.Module):
    """ResNet-based detector model architecture."""
    rx_size: int
    hidden_dim: int
    num_layers: int
    output_size: int
    activation: callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial projection to hidden dimension
        x = nn.Dense(self.hidden_dim)(x)
        x = self.activation(x)

        # ResNet blocks
        for _ in range(self.num_layers):
            x = ResNetBlock(
                hidden_dim=self.hidden_dim,
                activation=self.activation
            )(x, training=training)

        # Final projection to output size
        x = nn.Dense(self.output_size)(x)

        return x


class ResNetDetector(Detector):
    """ResNet-based detector.

    This is a non-modular implementation that processes all users together through
    a deep ResNet architecture with skip connections.

    Args:
        key (int | Array): Random key for parameter initialization.
        symbol_bits (int): Number of bits per symbol.
        num_users (int): Number of users.
        num_antennas (int): Number of receive antennas.
        num_layers (int): Number of ResNet layers.
        hidden_dim (int): Size of the hidden layers (later multiplied by log2(num_users)).
    """

    def __init__(
        self,
        key: int | Array,
        symbol_bits: int,
        num_users: int,
        num_antennas: int,
        num_layers: int,
        hidden_dim: int,
    ):
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim * num_users
        self.rx_size = 2 * num_antennas
        self.output_size = symbol_bits * num_users

        key = jr.PRNGKey(key) if isinstance(key, int) else key
        param_key, self.fit_key = jr.split(key)
        self._initialize_parameters(param_key)
        self.train_state = None

    def _initialize_parameters(self, key: Array):
        """Initialize the ResNet model parameters."""
        # Create the ResNet model
        self.model = ResNetDetectorModel(
            rx_size=self.rx_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_size=self.output_size
        )

        # Get parameter structure and apply function (same pattern as DeepSIC)
        dummy_input = jnp.empty((1, self.rx_size))
        first_params = self.model.init(key, dummy_input, training=False)
        self.params, unravel_fn = ravel_pytree(first_params)
        self.apply_fn = lambda w, x: self.model.apply(unravel_fn(w), x, training=False)
        self.param_size = len(self.params)

    def soft_decode(self, rx: Array) -> Array:
        """Soft-decode a (batch of) received signal(s).

        Args:
            rx (Array): Received signal(s).

        Returns:
            Array: Per user and per symbol-bit soft decisions.
        """
        predictions = self.apply_fn(self.params, rx)
        return jax.nn.sigmoid(predictions).reshape(rx.shape[0], self.num_users, self.symbol_bits)

    def classic_fit(self, rx: Array, labels: Array, state_init_fn: callable, train_block_fn: callable, **kwargs) -> None:
        """Fit model on all samples using the provided training function.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            state_init_fn (callable): Function to initialize the state.
            train_block_fn (callable): Training step for the entire model.
        """
        fit_key, self.fit_key = jr.split(self.fit_key)

        if self.train_state is None:
            self.train_state = state_init_fn(self.apply_fn, self.params)

        # Flatten labels to match the non-modular output shape
        labels_flat = labels.reshape(rx.shape[0], -1)

        # Train the model
        self.train_state, _ = train_block_fn(fit_key, self.train_state, rx, labels_flat)

        # Update the parameters
        self.params = self.train_state.params

    def streaming_fit(self, rx: Array, labels: Array, state_init_fn: callable, step_fn: callable, save_history: bool = False, **kwargs) -> Array:
        """Fit model on samples one by one.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            state_init_fn (callable): Function to initialize the state.
            step_fn (callable): Training step function.
            save_history (bool, optional): Whether to save and return the state history. Defaults to False.

        Returns:
            Array: State history if save_history is True, otherwise None.
        """
        if self.train_state is None:
            self.train_state = state_init_fn(self.apply_fn, self.params)

        # Flatten labels to match the non-modular output shape
        labels_flat = labels.reshape(rx.shape[0], -1)

        def process_sample(state, args):
            inputs, labels = args
            state, prediction = step_fn(state, inputs, labels)
            state_history = state if save_history else None
            return state, state_history

        self.train_state, state_history = jax.lax.scan(
            process_sample, 
            init=self.train_state, 
            xs=(rx, labels_flat)
        )

        self.params = self.train_state.params
        return state_history

    def save(self, path: str):
        """Save the model state to a file."""
        state = {
            'params': self.params,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """Load the model state from a file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.params = state['params']
        self.train_state = None
