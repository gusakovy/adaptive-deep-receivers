import pickle
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from adr.src.detectors.deepsic import DeepSICBlock
from adr.src.utils import CovarianceType


class BayesianDeepSIC:
    """Bayesian DeepSIC where each block is an independent Bayesian neural network.

    Args:
        key (int | Array): Random key for parameter initialization and training.
        symbol_bits (int): Number of bits per symbol.
        num_users (int): Number of users.
        num_antennas (int): Number of receive antennas.
        num_layers (int): Number of soft interference cancellation (SIC) layers.
        hidden_dim (int): Size of the hidden layer of each block.
        cov_type (CovarianceType, optional): Type of covariance for the parameters.
        init_cov_scale (float, optional): Initial parameter covariance scale. Defaults to 0.1.
    """

    def __init__(
        self,
        key: int | Array,
        symbol_bits: int,
        num_users: int,
        num_antennas: int,
        num_layers: int,
        hidden_dim: int,
        cov_type: CovarianceType = CovarianceType.FULL,
        init_cov_scale: float = 0.1,
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
        self.cov_type = cov_type
        self._initialize_parameters(param_key, init_cov_scale)

    def _initialize_parameters(self, key: Array, init_cov_scale: float):
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

        self.params_mean = jnp.stack(flat_params_list).reshape((self.num_layers, self.num_users, -1))

        # Initialize parameter covariances
        cov_shape = (self.param_size, self.param_size) if self.cov_type == CovarianceType.FULL else (self.param_size,)
        init_cov = init_cov_scale * jnp.eye(self.param_size) if self.cov_type == CovarianceType.FULL else init_cov_scale * jnp.ones(self.param_size)
        self.params_cov = jnp.tile(init_cov, (self.num_layers, self.num_users, 1, 1))
        self.params_cov = self.params_cov.reshape(self.num_layers, self.num_users, *cov_shape)

    def soft_decode(self, rx: Array) -> Array:
        """Soft-decode a (batch of) received signal(s).

        Args:
            rx (Array): Received signal(s).

        Returns:
            Array: Per user and per symbol-bit soft decisions.
        """
        def process_sample(rx_sample):
            """Process a single sample through all layers."""

            def process_sample_through_layer(carry, layer_params_mean):
                """Process a single sample through a single layer."""
                pred, rx_sample = carry

                layer_input = jnp.concatenate([pred, rx_sample], axis=-1)
                layer_pred = jax.vmap(self.apply_fn, in_axes=(0, None))(
                    layer_params_mean, layer_input
                )
                layer_pred = jax.nn.sigmoid(layer_pred)

                new_carry = (layer_pred.flatten(), rx_sample)
                return new_carry, layer_pred

            _, predictions = jax.lax.scan(
                process_sample_through_layer,
                init=(0.5 * jnp.ones(self.num_users * self.symbol_bits), rx_sample),
                xs=self.params_mean
            )
            return predictions[-1]

        predictions = jax.vmap(process_sample)(rx)
        return predictions

    def fit(self, rx: Array, labels: Array, step_fn: callable, **kwargs) -> Array:
        """Fit model on samples layer by layer. Each block is trained on all samples before moving on to the next layer.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            step_fn (callable): Training step function.
        """
        fit_key, self.fit_key = jr.split(self.fit_key)
        fit_keys = jr.split(fit_key, (self.num_layers, self.num_users, rx.shape[0]))

        def scannable_step_fn(carry, args):
            params_mean, params_cov = carry
            keys, inputs, labels = args
            new_params_mean, new_params_cov, prediction = step_fn(keys, params_mean, params_cov, inputs, labels)
            return (new_params_mean, new_params_cov), prediction

        def update_layer(carry, args):
            """Update a single layer."""
            pred, rx, labels = carry
            layer_keys, layer_params_mean, layer_params_cov = args

            layer_inputs = jnp.concatenate([pred, rx], axis=-1)

            def update_user_block(params_mean, params_cov, keys, inputs, labels):
                """Update a single user block."""
                (new_params_mean, new_params_cov), _ = jax.lax.scan(scannable_step_fn, init=(params_mean, params_cov), xs=(keys, inputs, labels))
                predictions = jax.nn.sigmoid(self.apply_fn(new_params_mean, inputs))
                return new_params_mean, new_params_cov, predictions

            new_params_mean, new_params_cov, predictions = jax.vmap(update_user_block, in_axes=(0, 0, 0, None, 1))(layer_params_mean, layer_params_cov, layer_keys, layer_inputs, labels)

            predictions = predictions.transpose(1, 0, 2).reshape(rx.shape[0], -1)
            new_carry = (predictions, rx, labels)
            output = (new_params_mean, new_params_cov)
            return new_carry, output

        initial_pred = 0.5 * jnp.ones((rx.shape[0], self.num_users * self.symbol_bits))
        _ , (self.params_mean, self.params_cov) = jax.lax.scan(update_layer, init=(initial_pred, rx, labels), xs=(fit_keys, self.params_mean, self.params_cov))

    def streaming_fit(self, rx: Array, labels: Array, step_fn: callable, save_history: bool = False, **kwargs) -> Array:
        """Fit model on samples one by one, processing each sample through all layers.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            step_fn (callable): Training step function.
            save_history (bool, optional): Whether to save and return the state history. Defaults to False.
        """
        fit_key, self.fit_key = jr.split(self.fit_key)
        fit_keys = jr.split(fit_key, (rx.shape[0], self.num_layers, self.num_users))

        def process_sample(state, args):
            """Process a single sample through all layers."""
            params_mean, params_cov = state
            keys, rx, labels = args

            def process_sample_through_layer(carry, args):
                """Process a single sample through a single layer."""
                pred, rx, labels = carry
                layer_keys, layer_params_mean, layer_params_cov = args

                layer_input = jnp.concatenate([pred, rx], axis=-1)
                layer_params_mean, layer_params_cov, layer_pred = jax.vmap(step_fn, in_axes=(0, 0, 0, None, 0))(
                    layer_keys, layer_params_mean, layer_params_cov, layer_input, labels
                )

                new_carry = (layer_pred.flatten(), rx, labels)
                output = (layer_params_mean, layer_params_cov)
                return new_carry, output

            _, new_state = jax.lax.scan(
                process_sample_through_layer, 
                init=(0.5 * jnp.ones(self.num_users * self.symbol_bits), rx, labels), 
                xs=(keys, params_mean, params_cov)
            )
            state_history = new_state if save_history else None
            return new_state, state_history

        final_state, state_history = jax.lax.scan(process_sample, init=(self.params_mean, self.params_cov), xs=(fit_keys, rx, labels))
        self.params_mean, self.params_cov = final_state
        return state_history

    def save(self, path: str):
        """Save the model state to a file."""
        state = {
            'params_mean': self.params_mean,
            'params_cov': self.params_cov
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """Load the model state from a file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.params_mean = state['params_mean']
        self.params_cov = state['params_cov']
