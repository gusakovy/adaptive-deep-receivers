import pytest
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import linen as nn
from adr import CovarianceType, DeepSIC, minibatch_sgd, iterative_ekf, fg_bong, dg_bong, fg_bog, dg_bog, fg_bbb, dg_bbb

@pytest.fixture
def setup_model(request):
    """Shared fixture for model initialization."""
    params = request.param
    symbol_bits = params.get("symbol_bits", 1)
    num_users = params.get("num_users", 1)
    num_antennas = params.get("num_antennas", 1)
    num_layers = params.get("num_layers", 1)
    hidden_dim = params.get("hidden_dim", 10)
    covariance_type = params.get("covariance_type", CovarianceType.NONE)

    model = DeepSIC(
        key=0,
        symbol_bits=symbol_bits,
        num_users=num_users,
        num_antennas=num_antennas,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        cov_type=covariance_type
    )
    return model


class TestSGD:
    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 1, "hidden_dim": 10},
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 4, "hidden_dim": 10}
    ], indirect=True)
    def test_minibatch_sgd_single_user_synthetic_data(self, setup_model):
        """Test single-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, 2))
        labels = jnp.expand_dims(jnp.where(jnp.sum(rx, axis=-1, keepdims=True) > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=minibatch_sgd,
            loss_fn=optax.sigmoid_binary_cross_entropy,
            num_epochs=5,
            batch_size=50,
            shuffle=True,
            optimizer=optax.adam(0.01),
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"

    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 1, "hidden_dim": 10},
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 4, "hidden_dim": 10}
    ], indirect=True)
    def test_minibatch_sgd_multi_user_synthetic_data(self, setup_model):
        """Test multi-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, setup_model.rx_size))
        labels = jnp.expand_dims(jnp.where(rx[:, :setup_model.num_antennas] + rx[:, setup_model.num_antennas:] > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=minibatch_sgd,
            loss_fn=optax.sigmoid_binary_cross_entropy,
            num_epochs=20,
            batch_size=50,
            shuffle=True,
            optimizer=optax.adam(0.01),
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"


class TestEKF:
    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 1, "hidden_dim": 10, "covariance_type": CovarianceType.FULL},
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 4, "hidden_dim": 10, "covariance_type": CovarianceType.FULL}
    ], indirect=True)
    def test_iterative_ekf_single_user_synthetic_data(self, setup_model):
        """Test single-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, 2))
        labels = jnp.expand_dims(jnp.where(jnp.sum(rx, axis=-1, keepdims=True) > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=iterative_ekf,
            init_param_cov=0.1,
            param_dynamics_function=lambda z, u: z,
            param_dynamics_cov=0.01,
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"

    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 1, "hidden_dim": 12, "covariance_type": CovarianceType.FULL},
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 4, "hidden_dim": 10, "covariance_type": CovarianceType.FULL}
    ], indirect=True)
    def test_iterative_ekf_multi_user_synthetic_data(self, setup_model):
        """Test multi-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, setup_model.rx_size))
        labels = jnp.expand_dims(jnp.where(rx[:, :setup_model.num_antennas] + rx[:, setup_model.num_antennas:] > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=iterative_ekf,
            init_param_cov=0.1,
            param_dynamics_function=lambda z, u: z,
            param_dynamics_cov=0.01,
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"


class TestBONG:
    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 1, "hidden_dim": 10, "covariance_type": CovarianceType.FULL},
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 4, "hidden_dim": 10, "covariance_type": CovarianceType.FULL}
    ], indirect=True)
    def test_fg_bong_single_user_synthetic_data(self, setup_model):
        """Test single-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, 2))
        labels = jnp.expand_dims(jnp.where(jnp.sum(rx, axis=-1, keepdims=True) > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=fg_bong,
            key=jr.PRNGKey(21),
            init_param_cov=0.1,
            log_likelihood=lambda mean, cov, y: -optax.sigmoid_binary_cross_entropy(mean, y),
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
            dynamics_decay=1.0,
            process_noise=0.0,
            num_samples=10,
            linplugin=True,
            empirical_cov=False,
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"

    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 1, "hidden_dim": 10, "covariance_type": CovarianceType.FULL},
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 4, "hidden_dim": 10, "covariance_type": CovarianceType.FULL}
    ], indirect=True)
    def test_fg_bong_multi_user_synthetic_data(self, setup_model):
        """Test multi-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, setup_model.rx_size))
        labels = jnp.expand_dims(jnp.where(rx[:, :setup_model.num_antennas] + rx[:, setup_model.num_antennas:] > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=fg_bong,
            key=jr.PRNGKey(21),
            init_param_cov=0.1,
            log_likelihood=lambda mean, cov, y: -optax.sigmoid_binary_cross_entropy(mean, y),
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
            dynamics_decay=1.0,
            process_noise=0.0,
            num_samples=10,
            linplugin=True,
            empirical_cov=False,
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"

    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 1, "hidden_dim": 10, "covariance_type": CovarianceType.DG},
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 4, "hidden_dim": 10, "covariance_type": CovarianceType.DG}
    ], indirect=True)
    def test_dg_bong_single_user_synthetic_data(self, setup_model):
        """Test single-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, 2))
        labels = jnp.expand_dims(jnp.where(jnp.sum(rx, axis=-1, keepdims=True) > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=dg_bong,
            key=jr.PRNGKey(21),
            init_param_cov=0.1,
            log_likelihood=lambda mean, cov, y: -optax.sigmoid_binary_cross_entropy(mean, y),
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
            dynamics_decay=1.0,
            process_noise=0.0,
            num_samples=10,
            linplugin=True,
            empirical_cov=False,
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"

    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 1, "hidden_dim": 10, "covariance_type": CovarianceType.DG},
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 4, "hidden_dim": 10, "covariance_type": CovarianceType.DG}
    ], indirect=True)
    def test_dg_bong_multi_user_synthetic_data(self, setup_model):
        """Test multi-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, setup_model.rx_size))
        labels = jnp.expand_dims(jnp.where(rx[:, :setup_model.num_antennas] + rx[:, setup_model.num_antennas:] > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=dg_bong,
            key=jr.PRNGKey(21),
            init_param_cov=0.1,
            log_likelihood=lambda mean, cov, y: -optax.sigmoid_binary_cross_entropy(mean, y),
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
            dynamics_decay=1.0,
            process_noise=0.0,
            num_samples=10,
            linplugin=True,
            empirical_cov=False,
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"


class TestBOG:
    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 1, "hidden_dim": 10, "covariance_type": CovarianceType.FULL}
    ], indirect=True)
    def test_fg_bog_single_user_synthetic_data(self, setup_model):
        """Test single-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, 2))
        labels = jnp.expand_dims(jnp.where(jnp.sum(rx, axis=-1, keepdims=True) > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=fg_bog,
            key=jr.PRNGKey(21),
            init_param_cov=0.1,
            log_likelihood=lambda mean, cov, y: -optax.sigmoid_binary_cross_entropy(mean, y),
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
            dynamics_decay=1.0,
            process_noise=0.0,
            num_samples=10,
            linplugin=True,
            empirical_fisher=False,
            learning_rate=0.2,
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"

    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 1, "hidden_dim": 10, "covariance_type": CovarianceType.DG}
    ], indirect=True)
    def test_dg_bog_single_user_synthetic_data(self, setup_model):
        """Test single-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, 2))
        labels = jnp.expand_dims(jnp.where(jnp.sum(rx, axis=-1, keepdims=True) > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=dg_bog,
            key=jr.PRNGKey(21),
            init_param_cov=0.1,
            log_likelihood=lambda mean, cov, y: -optax.sigmoid_binary_cross_entropy(mean, y),
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
            dynamics_decay=1.0,
            process_noise=0.0,
            num_samples=10,
            linplugin=True,
            empirical_fisher=False,
            learning_rate=0.2,
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"


class TestBBB:
    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 1, "hidden_dim": 10, "covariance_type": CovarianceType.FULL}
    ], indirect=True)
    def test_fg_bbb_single_user_synthetic_data(self, setup_model):
        """Test single-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, 2))
        labels = jnp.expand_dims(jnp.where(jnp.sum(rx, axis=-1, keepdims=True) > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=fg_bbb,
            key=jr.PRNGKey(21),
            init_param_cov=0.1,
            log_likelihood=lambda mean, cov, y: -optax.sigmoid_binary_cross_entropy(mean, y),
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
            dynamics_decay=1.0,
            process_noise=0.0,
            num_samples=10,
            linplugin=True,
            empirical_fisher=False,
            learning_rate=0.1,
            num_iter=10
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"

    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 1, "hidden_dim": 10, "covariance_type": CovarianceType.DG}
    ], indirect=True)
    def test_dg_bbb_single_user_synthetic_data(self, setup_model):
        """Test single-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, 2))
        labels = jnp.expand_dims(jnp.where(jnp.sum(rx, axis=-1, keepdims=True) > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=dg_bbb,
            key=jr.PRNGKey(21),
            init_param_cov=0.1,
            log_likelihood=lambda mean, cov, y: -optax.sigmoid_binary_cross_entropy(mean, y),
            obs_function=nn.sigmoid,
            obs_cov=0.1*jnp.eye(1),
            dynamics_decay=1.0,
            process_noise=0.0,
            num_samples=10,
            linplugin=True,
            empirical_fisher=False,
            learning_rate=0.1,
            num_iter=10
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"
