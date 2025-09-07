import pickle
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from flax import linen as nn
from adr.src.detectors.base import Detector


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
        self.hidden_dim = hidden_dim
        self.rx_size = 2 * num_antennas
        self.block_input_size = self.rx_size + symbol_bits * num_users

        key = jr.PRNGKey(key) if isinstance(key, int) else key
        param_key, self.fit_key = jr.split(key)
        self._initialize_parameters(param_key)
        self.train_state = None

    def _initialize_parameters(self, key: Array):
        """Initialize parameter means and covariances."""
        # Initialize block model
        block_model = DeepSICBlock(
            symbol_bits=self.symbol_bits,
            num_users=self.num_users,
            num_antennas=self.num_antennas,
            hidden_dim=self.hidden_dim
        )
        subkeys = jr.split(key, self.num_users * self.num_layers)

        # Get parameter structure and apply function
        first_params = block_model.init(subkeys[0], jnp.empty((1, self.block_input_size)))
        flat_params, unravel_fn = ravel_pytree(first_params)
        self.apply_fn = lambda w, x: block_model.apply(unravel_fn(w), x)
        self.param_size = len(flat_params)

        # Initialize parameter means
        flat_params_list = []
        for layer_idx in range(self.num_layers):
            for user_idx in range(self.num_users):
                subkey = subkeys[layer_idx * self.num_users + user_idx]
                params = block_model.init(subkey, jnp.empty((1, self.block_input_size)))
                flat_params, _ = ravel_pytree(params)
                flat_params_list.append(flat_params)

        self.params = jnp.stack(flat_params_list).reshape((self.num_layers, self.num_users, -1))

    def soft_decode(self, rx: Array) -> Array:
        """Soft-decode a (batch of) received signal(s).

        Args:
            rx (Array): Received signal(s).

        Returns:
            Array: Per user and per symbol-bit soft decisions.
        """
        def process_sample(rx_sample):
            """Process a single sample through all layers."""

            def process_sample_through_layer(carry, layer_params):
                """Process a single sample through a single layer."""
                pred, rx_sample = carry

                layer_input = jnp.concatenate([pred, rx_sample], axis=-1)
                layer_pred = jax.vmap(self.apply_fn, in_axes=(0, None))(
                    layer_params, layer_input
                )
                layer_pred = jax.nn.sigmoid(layer_pred)

                new_carry = (layer_pred.flatten(), rx_sample)
                return new_carry, layer_pred

            _, predictions = jax.lax.scan(
                process_sample_through_layer,
                init=(0.5 * jnp.ones(self.num_users * self.symbol_bits), rx_sample),
                xs=self.params
            )
            return predictions[-1]

        predictions = jax.vmap(process_sample)(rx)
        return predictions

    def classic_fit(self, rx: Array, labels: Array, state_init_fn: callable, extract_params: callable, train_block_fn: callable, **kwargs) -> None:
        """Train each block independently using the provided train_block_fn method.
        Training is performed layer-by-layer, with parallel updates for user blocks within each layer.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            state_init_fn (callable): Function to initialize the state.
            extract_params (callable): Function to extract the parameters from the state.
            train_block_fn (callable): Function to train a single block.
        """
        fit_key, self.fit_key = jr.split(self.fit_key)
        fit_keys = jr.split(fit_key, (self.num_layers, self.num_users))

        if self.train_state is None:
            init_fn = lambda params: state_init_fn(self.apply_fn, params)
            self.train_state = jax.vmap(jax.vmap(init_fn))(self.params)

        def train_layer(carry, args):
            """Update a single layer."""
            pred, rx, labels = carry
            layer_keys, layer_states = args

            layer_inputs = jnp.concatenate([pred, rx], axis=-1)

            new_states, outputs = jax.vmap(
                train_block_fn, 
                in_axes=(0, 0, None, 1)
            )(layer_keys, layer_states, layer_inputs, labels)

            predictions = jax.nn.sigmoid(outputs)
            predictions = predictions.transpose(1, 0, 2).reshape(rx.shape[0], -1)
            new_carry = (predictions, rx, labels)
            return new_carry, new_states

        initial_pred = 0.5 * jnp.ones((rx.shape[0], self.num_users * self.symbol_bits))
        _, self.train_state = jax.lax.scan(
            train_layer, 
            init=(initial_pred, rx, labels), 
            xs=(fit_keys, self.train_state)
        )

        self.params = extract_params(self.train_state)

    def fit(self, rx: Array, labels: Array, state_init_fn: callable, extract_params: callable, step_fn: callable, **kwargs) -> Array:
        """Fit model on samples layer by layer. Each block is trained on all samples before moving on to the next layer.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            state_init_fn (callable): Function to initialize the state.
            extract_params (callable): Function to extract the parameters from the state.
            step_fn (callable): Training step function.
        """
        fit_key, self.fit_key = jr.split(self.fit_key)
        fit_keys = jr.split(fit_key, (self.num_layers, self.num_users, rx.shape[0]))
    
        if self.train_state is None:
            init_fn = lambda params: state_init_fn(self.apply_fn, params)
            self.train_state = jax.vmap(jax.vmap(init_fn))(self.params)

        def scannable_step_fn(state, args):
            key, inputs, labels = args
            state, prediction = step_fn(key, state, inputs, labels)
            return state, prediction

        def update_layer(carry, args):
            """Update a single layer."""
            pred, rx, labels = carry
            layer_keys, layer_states = args

            layer_inputs = jnp.concatenate([pred, rx], axis=-1)

            def update_user_block(key, state, inputs, labels):
                """Update a single user block."""
                state, _ = jax.lax.scan(scannable_step_fn, init=state, xs=(inputs, labels))
                predictions = jax.nn.sigmoid(self.apply_fn(extract_params(state), inputs))
                return state, predictions

            new_state, predictions = jax.vmap(update_user_block, in_axes=(0, 0, None, 1))(layer_keys, layer_states, layer_inputs, labels)

            predictions = predictions.transpose(1, 0, 2).reshape(rx.shape[0], -1)
            new_carry = (predictions, rx, labels)
            return new_carry, new_state

        initial_pred = 0.5 * jnp.ones((rx.shape[0], self.num_users * self.symbol_bits))
        _ , self.train_state = jax.lax.scan(update_layer, init=(initial_pred, rx, labels), xs=(fit_keys, self.train_state))
        self.params = extract_params(self.train_state)

    def streaming_fit(self, rx: Array, labels: Array, state_init_fn: callable, extract_params: callable, step_fn: callable, save_history: bool = False, **kwargs) -> Array:
        """Fit model on samples one by one, processing each sample through all layers.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            state_init_fn (callable): Function to initialize the state.
            extract_params (callable): Function to extract the parameters from the state.
            step_fn (callable): Training step function.
            save_history (bool, optional): Whether to save and return the state history. Defaults to False.

        Returns:
            Array: State history if save_history is True, otherwise None.
        """
        fit_key, self.fit_key = jr.split(self.fit_key)
        fit_keys = jr.split(fit_key, (rx.shape[0], self.num_layers, self.num_users))

        if self.train_state is None:
            init_fn = lambda params: state_init_fn(self.apply_fn, params)
            self.train_state = jax.vmap(jax.vmap(init_fn))(self.params)

        def process_sample(state, args):
            """Process a single sample through all layers."""
            keys, rx, labels = args

            def process_sample_through_layer(carry, args):
                """Process a single sample through a single layer."""
                pred, rx, labels = carry
                layer_keys, state = args

                layer_input = jnp.concatenate([pred, rx], axis=-1)
                state, layer_pred = jax.vmap(step_fn, in_axes=(0, 0, None, 0))(
                    layer_keys, state, layer_input, labels
                )

                new_carry = (layer_pred.flatten(), rx, labels)
                return new_carry, state

            _, new_state = jax.lax.scan(
                process_sample_through_layer,
                init=(0.5 * jnp.ones(self.num_users * self.symbol_bits), rx, labels),
                xs=(keys, state)
            )
            state_history = new_state if save_history else None
            return new_state, state_history

        self.train_state, state_history = jax.lax.scan(process_sample, init=self.train_state, xs=(fit_keys, rx, labels))
        self.params = extract_params(self.train_state)
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
