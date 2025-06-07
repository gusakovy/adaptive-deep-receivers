import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from adr.src.detectors.base import Detector
from adr.src.detectors.deepsic_block import DeepSICBlock

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

    def layer_transition(self, layer_num: int, rx: Array, pred: Array = None) -> Array:
        """Pass data through a layer.

        Args:
            layer_num (int): Layer number.
            rx (Array): Received signal(s).
            pred (Array, optional): Soft decisions from the previous layer. Defaults to None.

        Returns:
            Array: Per symbol-bit soft decisions for all blocks in the layer.
        """
        def block_soft_decode(params, inputs):
            bitwise_logits = self.apply_fn(params, inputs)
            return jax.nn.sigmoid(bitwise_logits)

        inputs = self._pred_and_rx_to_input(layer_num, rx, pred)
        layer_outputs = jax.vmap(block_soft_decode, in_axes=(0, None))(self.params[layer_num], inputs)

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

    def classic_fit(self, rx: Array, labels: Array, state_init_fn: callable, train_block_fn: callable, **kwargs) -> None:
        """Train each block independently using the provided train_block_fn method.
        Training is performed layer-by-layer, with parallel updates for user blocks within each layer.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            state_init_fn (callable): Function to initialize the state.
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

        self.params = self.train_state.params

    def fit(self, rx: Array, labels: Array, state_init_fn: callable, step_fn: callable, **kwargs) -> Array:
        """Fit model on samples layer by layer. Each block is trained on all samples before moving on to the next layer.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            state_init_fn (callable): Function to initialize the state.
            step_fn (callable): Training step function.
        """
        if self.train_state is None:
            init_fn = lambda params: state_init_fn(self.apply_fn, params)
            self.train_state = jax.vmap(jax.vmap(init_fn))(self.params)

        def scannable_step_fn(state, args):
            inputs, labels = args
            state, prediction = step_fn(state, inputs, labels)
            return state, prediction

        def update_layer(carry, state):
            """Update a single layer."""
            pred, rx, labels = carry

            layer_inputs = jnp.concatenate([pred, rx], axis=-1)

            def update_user_block(state, inputs, labels):
                """Update a single user block."""
                state, _ = jax.lax.scan(scannable_step_fn, init=state, xs=(inputs, labels))
                predictions = jax.nn.sigmoid(self.apply_fn(state.params, inputs))
                return state, predictions

            new_state, predictions = jax.vmap(update_user_block, in_axes=(0, None, 1))(state, layer_inputs, labels)

            predictions = predictions.transpose(1, 0, 2).reshape(rx.shape[0], -1)
            new_carry = (predictions, rx, labels)
            return new_carry, new_state

        initial_pred = 0.5 * jnp.ones((rx.shape[0], self.num_users * self.symbol_bits))
        _ , self.train_state = jax.lax.scan(update_layer, init=(initial_pred, rx, labels), xs=self.train_state)
        self.params = self.train_state.params

    def streaming_fit(self, rx: Array, labels: Array, state_init_fn: callable, step_fn: callable, save_history: bool = False, **kwargs) -> Array:
        """Fit model on samples one by one, processing each sample through all layers.

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
            init_fn = lambda params: state_init_fn(self.apply_fn, params)
            self.train_state = jax.vmap(jax.vmap(init_fn))(self.params)

        def process_sample(state, args):
            """Process a single sample through all layers."""
            rx, labels = args

            def process_sample_through_layer(carry, state):
                """Process a single sample through a single layer."""
                pred, rx, labels = carry

                layer_input = jnp.concatenate([pred, rx], axis=-1)
                state, layer_pred = jax.vmap(step_fn, in_axes=(0, None, 0))(
                    state, layer_input, labels
                )

                new_carry = (layer_pred.flatten(), rx, labels)
                return new_carry, state

            _, new_state = jax.lax.scan(
                process_sample_through_layer,
                init=(0.5 * jnp.ones(self.num_users * self.symbol_bits), rx, labels),
                xs=state
            )
            state_history = new_state if save_history else None
            return new_state, state_history

        self.train_state, state_history = jax.lax.scan(process_sample, init=self.train_state, xs=(rx, labels))
        self.params = self.train_state.params
        return state_history

    def save(self, path: str):
        """Save the model state to a file."""
        jnp.save(path, self.params)

    def load(self, path: str):
        """Load the model state from a file."""
        self.params = jnp.load(path)
