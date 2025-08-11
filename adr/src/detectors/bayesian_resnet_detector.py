import pickle
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from adr.src.utils import CovarianceType
from adr.src.detectors.base import Detector
from adr.src.detectors.resnet_detector import ResNetDetectorModel


class BayesianResNetDetector(Detector):
    """Bayesian ResNet-based detector with similar parameter count to original Bayesian DeepSIC.

    This is a non-modular implementation that processes all users together through
    a ResNet architecture with skip connections, where the parameters are treated
    as Bayesian variables.

    Args:
        key (int | Array): Random key for parameter initialization and training.
        symbol_bits (int): Number of bits per symbol.
        num_users (int): Number of users.
        num_antennas (int): Number of receive antennas.
        num_layers (int): Number of ResNet layers.
        hidden_dim (int): Size of the hidden layers.
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
        self.hidden_dim = hidden_dim * num_users
        self.rx_size = 2 * num_antennas
        self.output_size = symbol_bits * num_users

        key = jr.PRNGKey(key) if isinstance(key, int) else key
        param_key, self.fit_key = jr.split(key)
        self.cov_type = cov_type
        self._initialize_parameters(param_key, init_cov_scale)

    def _initialize_parameters(self, key: Array, init_cov_scale: float):
        """Initialize parameter means and covariances."""
        # Create the ResNet model
        self.model = ResNetDetectorModel(
            rx_size=self.rx_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_size=self.output_size
        )

        # Get parameter structure and apply function (same pattern as BayesianDeepSIC)
        dummy_input = jnp.empty((1, self.rx_size))
        first_params = self.model.init(key, dummy_input, training=False)
        flat_params, unravel_fn = ravel_pytree(first_params)
        self.apply_fn = lambda w, x: self.model.apply(unravel_fn(w), x, training=False)
        self.param_size = len(flat_params)

        # Initialize parameter means
        self.params_mean = flat_params

        # Initialize parameter covariances
        if self.cov_type == CovarianceType.FULL:
            self.params_cov = init_cov_scale * jnp.eye(self.param_size)
        else:  # CovarianceType.DG
            self.params_cov = init_cov_scale * jnp.ones(self.param_size)

    def soft_decode(self, rx: Array) -> Array:
        """Soft-decode a (batch of) received signal(s).

        Args:
            rx (Array): Received signal(s).

        Returns:
            Array: Per user and per symbol-bit soft decisions.
        """
        predictions = self.apply_fn(self.params_mean, rx)
        return jax.nn.sigmoid(predictions).reshape(rx.shape[0], self.num_users, self.symbol_bits)

    def streaming_fit(self, rx: Array, labels: Array, step_fn: callable, save_history: bool = False, **kwargs) -> Array:
        """Fit model on samples one by one.

        Args:
            rx (Array): Received signal(s).
            labels (Array): Bitwise labels corresponding to the received signal(s).
            step_fn (callable): Training step function.
            save_history (bool, optional): Whether to save and return the state history. Defaults to False.

        Returns:
            Array: State history if save_history is True, otherwise None.
        """
        fit_key, self.fit_key = jr.split(self.fit_key)
        fit_keys = jr.split(fit_key, rx.shape[0])

        def process_sample(state, args):
            params_mean, params_cov = state
            key, inputs, labels = args
            new_params_mean, new_params_cov, prediction = step_fn(key, params_mean, params_cov, inputs, labels)
            new_state = (new_params_mean, new_params_cov)
            state_history = new_state if save_history else None
            return new_state, state_history

        # Flatten labels to match the non-modular output shape
        labels_flat = labels.reshape(rx.shape[0], -1)

        final_state, state_history = jax.lax.scan(
            process_sample,
            init=(self.params_mean, self.params_cov),
            xs=(fit_keys, rx, labels_flat)
        )

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
