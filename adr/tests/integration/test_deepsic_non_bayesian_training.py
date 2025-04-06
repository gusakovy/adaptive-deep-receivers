import pytest
import jax.numpy as jnp
import jax.random as jr
import optax
from adr import DeepSIC, minibatch_sgd

@pytest.fixture
def setup_model(request):
    """Shared fixture for model initialization."""
    params = request.param
    symbol_bits = params.get("symbol_bits", 1)
    num_users = params.get("num_users", 1)
    num_antennas = params.get("num_antennas", 1)
    num_layers = params.get("num_layers", 1)
    hidden_dim = params.get("hidden_dim", 10)

    model = DeepSIC(
        key=0,
        symbol_bits=symbol_bits,
        num_users=num_users,
        num_antennas=num_antennas,
        num_layers=num_layers,
        hidden_dim=hidden_dim
    )
    return model


class TestMinibatchSGD:
    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 1, "hidden_dim": 10},
        {"symbol_bits": 1, "num_users": 1, "num_antennas": 1, "num_layers": 4, "hidden_dim": 10}
    ], indirect=True)
    def test_single_user_synthetic_data(self, setup_model):
        """Test single-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, 1))
        labels = jnp.expand_dims(jnp.where(rx > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=minibatch_sgd,
            loss_fn=optax.sigmoid_binary_cross_entropy,
            num_epochs=5,
            batch_size=50,
            shuffle=True
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"

    @pytest.mark.parametrize("setup_model", [
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 1, "hidden_dim": 10},
        {"symbol_bits": 1, "num_users": 4, "num_antennas": 4, "num_layers": 4, "hidden_dim": 10}
    ], indirect=True)
    def test_multi_user_synthetic_data(self, setup_model):
        """Test multi-user training process with synthetic data."""
        key = jr.PRNGKey(42)
        data_size = 1000
        rx = jr.normal(key, (data_size, setup_model.rx_size))
        labels = jnp.expand_dims(jnp.where(rx > 0, 1.0, 0.0), axis=-1)

        setup_model.fit(
            rx=rx, 
            labels=labels, 
            train_block_fn=minibatch_sgd,
            loss_fn=optax.sigmoid_binary_cross_entropy,
            num_epochs=20,
            batch_size=50,
            shuffle=True,
        )

        predictions = setup_model.soft_decode(rx)
        predicted_labels = (predictions > 0.5).astype(jnp.float32)
        accuracy = (predicted_labels == labels).mean()

        assert accuracy > 0.95, f"Expected accuracy > 0.95, got {accuracy}"
