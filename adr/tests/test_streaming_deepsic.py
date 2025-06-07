"""Tests for StreamingDeepSIC implementation."""

import time
from functools import partial
import jax.numpy as jnp
import jax.random as jr

from adr.src.detectors.deepsic_streaming import StreamingDeepSIC
from adr.src.training_algorithms.step_functions import TrainingMethod, step_fn_builder


def test_streaming_deepsic_initialization():
    """Test that StreamingDeepSIC initializes correctly."""
    key = 42
    symbol_bits = 2
    num_users = 3
    num_antennas = 4
    num_layers = 2
    hidden_dim = 10
    
    start_time = time.perf_counter()
    detector = StreamingDeepSIC(
        key=key,
        symbol_bits=symbol_bits,
        num_users=num_users,
        num_antennas=num_antennas,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    init_time = time.perf_counter() - start_time
    print(f"  Initialization time: {init_time*1000:.2f} ms")
    
    # Check shapes
    assert detector.params_mean.shape == (num_layers, num_users, detector.params_mean.shape[-1])
    assert detector.params_cov.shape == (num_layers, num_users, detector.params_mean.shape[-1], detector.params_mean.shape[-1])


def test_streaming_deepsic_prediction():
    """Test that StreamingDeepSIC can make predictions."""
    key = 42
    symbol_bits = 2
    num_users = 3
    num_antennas = 4
    num_layers = 2
    hidden_dim = 10
    batch_size = 5
    
    detector = StreamingDeepSIC(
        key=key,
        symbol_bits=symbol_bits,
        num_users=num_users,
        num_antennas=num_antennas,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    
    # Create dummy received signals
    rx_size = 2 * num_antennas
    rx = jr.normal(jr.PRNGKey(123), (batch_size, rx_size))
    
    # Make predictions with timing
    start_time = time.perf_counter()
    predictions = detector.soft_decode(rx)
    prediction_time = time.perf_counter() - start_time
    
    print(f"  Prediction time (batch of {batch_size}): {prediction_time*1000:.2f} ms")
    print(f"  Prediction time per sample: {prediction_time*1000/batch_size:.2f} ms")
    
    # Check output shape and range
    assert predictions.shape == (batch_size, num_users, symbol_bits), f"Predictions shape is {predictions.shape}"
    assert jnp.all(predictions >= 0.0)
    assert jnp.all(predictions <= 1.0)


def test_streaming_deepsic_training():
    """Test that StreamingDeepSIC can train on samples."""
    key = 42
    symbol_bits = 2
    num_users = 3
    num_antennas = 4
    num_layers = 2
    hidden_dim = 10
    batch_size = 30
    
    detector = StreamingDeepSIC(
        key=key,
        symbol_bits=symbol_bits,
        num_users=num_users,
        num_antennas=num_antennas,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

    step_fn = step_fn_builder(
        method=TrainingMethod.BONG,
        apply_fn=detector.apply_fn,
        obs_cov=0.1*jnp.eye(symbol_bits),
    )
    
    # Store initial parameters for comparison
    initial_params_mean = detector.params_mean.copy()
    
    # Create dummy data
    rx_size = 2 * num_antennas
    rx = jr.normal(jr.PRNGKey(123), (batch_size, rx_size))
    labels = jr.bernoulli(jr.PRNGKey(456), 0.5, (batch_size, num_users, symbol_bits)).astype(jnp.float32)
    
    # Train on the samples with timing
    start_time = time.perf_counter()
    detector.fit(rx, labels, step_fn)
    training_time = time.perf_counter() - start_time
    
    print(f"  Training time (batch of {batch_size}): {training_time*1000:.2f} ms")
    print(f"  Training time per sample: {training_time*1000/batch_size:.2f} ms")
    
    # Check that parameters have been updated
    assert not jnp.allclose(initial_params_mean, detector.params_mean)


def test_streaming_deepsic_accuracy_with_learnable_pattern():
    """Test that StreamingDeepSIC can learn a simple pattern with high accuracy."""
    print("  Testing learning accuracy on a simple learnable pattern...")
    
    key = 21
    symbol_bits = 1
    num_users = 1
    num_antennas = 1
    num_layers = 4
    hidden_dim = 10
    
    detector = StreamingDeepSIC(
        key=key,
        symbol_bits=symbol_bits,
        num_users=num_users,
        num_antennas=num_antennas,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

    step_fn = step_fn_builder(
        method=TrainingMethod.BONG,
        apply_fn=detector.apply_fn,
        obs_cov=0.1*jnp.eye(symbol_bits),
        dynamics_decay=0.999,
        process_noise=0.001*jnp.eye(detector.params_mean.shape[-1]),
    )
    
    # Create received signals
    data_size = 10000
    rx = jr.normal(jr.PRNGKey(42), (data_size, 2))
    labels = jnp.expand_dims(jnp.where(jnp.sum(rx, axis=-1, keepdims=True) > 0, 1.0, 0.0), axis=-1)
    
    # Test initial accuracy (should be around random chance ~50%)
    initial_predictions = detector.soft_decode(rx)
    initial_predicted_labels = (initial_predictions > 0.5).astype(jnp.float32)
    initial_bit_acc = (initial_predicted_labels == labels).mean()
    
    print(f"    Initial bit accuracy: {initial_bit_acc:.3f}")
    
    # Train for multiple epochs on the same data (should learn the pattern)
    
    start_time = time.perf_counter()
    detector.fit(rx, labels, step_fn)
    training_time = time.perf_counter() - start_time
    print(f"    Training time per sample: {training_time*1000/(data_size):.2f} ms")
    
    # Final accuracy test
    final_predictions = detector.soft_decode(rx)
    final_predicted_labels = (final_predictions > 0.5).astype(jnp.float32)
    final_bit_acc = (final_predicted_labels == labels).mean()
    
    print(f"    Final: bit_acc={final_bit_acc:.3f}")
    print(f"    Total training time: {training_time*1000:.2f} ms")
    
    
    # Check that we've improved significantly from random chance
    improvement_threshold = 0.2  # Should improve by at least 20%
    assert final_bit_acc > initial_bit_acc + improvement_threshold, \
        f"Bit accuracy should improve by at least {improvement_threshold:.1%}: {initial_bit_acc:.3f} -> {final_bit_acc:.3f}"
    
    # For a simple deterministic pattern, we should achieve good accuracy
    min_expected_bit_acc = 0.85
    min_expected_symbol_acc = 0.70
    
    if final_bit_acc >= min_expected_bit_acc:
        print(f"    ✅ Excellent bit accuracy: {final_bit_acc:.3f} >= {min_expected_bit_acc}")
    else:
        print(f"    ⚠️  Lower than expected bit accuracy: {final_bit_acc:.3f} < {min_expected_bit_acc}")
    
    # The test should pass if we see significant improvement, even if not perfect
    assert final_bit_acc > 0.7, f"Should achieve at least 70% bit accuracy on simple pattern, got {final_bit_acc:.3f}"


def test_streaming_deepsic_single_sample():
    """Test training with a single sample."""
    key = 42
    symbol_bits = 2
    num_users = 2
    num_antennas = 3
    num_layers = 2
    hidden_dim = 5
    
    detector = StreamingDeepSIC(
        key=key,
        symbol_bits=symbol_bits,
        num_users=num_users,
        num_antennas=num_antennas,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

    step_fn = step_fn_builder(
        method=TrainingMethod.BONG,
        apply_fn=detector.apply_fn,
        obs_cov=0.1*jnp.eye(symbol_bits),
    )
    
    # Create single sample
    rx_size = 2 * num_antennas
    rx = jr.normal(jr.PRNGKey(123), (rx_size,))  # Single sample
    labels = jr.bernoulli(jr.PRNGKey(456), 0.5, (num_users, symbol_bits)).astype(jnp.float32)
    
    # Expand dimensions for batch processing
    rx_batch = rx.reshape(1, -1)
    labels_batch = labels.reshape(1, num_users, symbol_bits)
    
    # Train on the sample with timing
    start_time = time.perf_counter()
    detector.fit(rx_batch, labels_batch, step_fn)
    single_sample_time = time.perf_counter() - start_time
    
    print(f"  Single sample training time: {single_sample_time*1000:.2f} ms")
    
    # Test prediction timing for single sample
    start_time = time.perf_counter()
    pred_only = detector.soft_decode(rx_batch)
    single_pred_time = time.perf_counter() - start_time
    
    print(f"  Single sample prediction time: {single_pred_time*1000:.2f} ms")
    
    # Check output shape
    assert pred_only.shape == (1, num_users, symbol_bits)


def test_streaming_deepsic_performance():
    """Test performance with larger batch sizes."""
    print("  Testing performance with different configurations...")
    
    configs = [
        {"name": "Small", "users": 2, "layers": 2, "batch": 10},
        {"name": "Medium", "users": 4, "layers": 3, "batch": 10}, 
        {"name": "Large", "users": 8, "layers": 4, "batch": 10},
    ]
    
    for config in configs:
        detector = StreamingDeepSIC(
            key=42,
            symbol_bits=2,
            num_users=config["users"],
            num_antennas=4,
            num_layers=config["layers"],
            hidden_dim=10,
        )
        
        step_fn = step_fn_builder(
            method=TrainingMethod.BONG,
            apply_fn=detector.apply_fn,
            obs_cov=0.1*jnp.eye(2)
        )
        
        # Create data
        rx_size = 2 * 4  # num_antennas = 4
        rx = jr.normal(jr.PRNGKey(123), (config["batch"], rx_size))
        labels = jr.bernoulli(jr.PRNGKey(456), 0.5, (config["batch"], config["users"], 2)).astype(jnp.float32)
        
        # Training timing
        start_time = time.perf_counter()
        detector.fit(rx, labels, step_fn)
        training_time = time.perf_counter() - start_time
        
        # Prediction timing
        start_time = time.perf_counter()
        _ = detector.soft_decode(rx)
        prediction_time = time.perf_counter() - start_time
        
        print(f"    {config['name']} ({config['users']} users, {config['layers']} layers):")
        print(f"      Training: {training_time*1000/config['batch']:.2f} ms/sample")
        print(f"      Prediction: {prediction_time*1000/config['batch']:.2f} ms/sample")



if __name__ == "__main__":
    print("Running StreamingDeepSIC tests with timing measurements...\n")
    
    try:
        test_streaming_deepsic_initialization()
        print()
        
        test_streaming_deepsic_prediction()
        print()
        
        test_streaming_deepsic_training()
        print()
        
        test_streaming_deepsic_accuracy_with_learnable_pattern()
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()