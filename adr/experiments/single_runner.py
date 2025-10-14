import time
import os
import math
from copy import deepcopy
import json
import pickle
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import optax
from adr import Detector, DeepSIC, ResNetDetector
from adr import CovarianceType, TrainingMethod, UplinkMimoChannel
from adr import step_fn_builder, build_gd_step_fn, build_sgd_train_fn
from adr.src.channels.modulations import MODULATIONS
from adr.experiments.utils import load_config, generate_config_hash, prepare_experiment_data, prepare_single_batch


COV_TYPE_MAP = {
    'full': CovarianceType.FULL,
    'dlr': CovarianceType.DLR,
    'diag': CovarianceType.DG
}

METHOD_MAP = {
    'gd': TrainingMethod.GD,
    'sgd': TrainingMethod.SGD,
    'bbb': TrainingMethod.BBB,
    'blr': TrainingMethod.BLR,
    'bog': TrainingMethod.BOG,
    'bong': TrainingMethod.BONG 
}

def validate_config(config: dict[str, any]) -> None:
    """Validate that the config contains all required sections and parameters."""
    required_sections = ['model', 'channel', 'experiment', 'algorithm']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    algo_required = ['method', 'dynamics_decay']
    for param in algo_required:
        if param not in config['algorithm']:
            raise ValueError(f"Missing required algorithm parameter: {param}")

    model_required = ['type', 'num_layers', 'hidden_dim']
    for param in model_required:
        if param not in config['model']:
            raise ValueError(f"Missing required model parameter: {param}")

    channel_required = ['modulation', 'snr', 'num_users', 'num_antennas']
    for param in channel_required:
        if param not in config['channel']:
            raise ValueError(f"Missing required channel parameter: {param}")

    exp_required = ['sync_frames', 'track_frames', 'symbols_per_frame', 'pilot_per_frame', 'test_dim', 'seed']
    for param in exp_required:
        if param not in config['experiment']:
            raise ValueError(f"Missing required experiment parameter: {param}")

def clean_config(config: dict[str, any]) -> dict[str, any]:
    """Remove unused parameters based on the method type."""
    method = config['algorithm']['method'].lower()
    model_type = config['model']['type'].lower()
    cleaned_config = deepcopy(config)

    # Method-specific parameter requirements
    method_params = {
        'gd': {
            'required': {'learning_rate', 'num_iter'},
            'unused': {'covariance_type', 'linplugin', 'reparameterized', 'obs_cov_scale', 'process_noise', 'num_samples', 'empirical_fisher', 'batch_size'}
        },
        'sgd': {
            'required': {'learning_rate', 'num_iter', 'batch_size'},
            'unused': {'covariance_type', 'linplugin', 'reparameterized', 'obs_cov_scale', 'process_noise', 'num_samples', 'empirical_fisher'}
        },
        'bbb': {
            'required': {'covariance_type', 'learning_rate', 'num_iter'},
            'unused': {'batch_size'}
        },
        'blr': {
            'required': {'covariance_type', 'learning_rate', 'num_iter'},
            'unused': {'batch_size'}
        },
        'bog': {
            'required': {'covariance_type', 'learning_rate', 'num_iter'},
            'unused': {'batch_size'}
        },
        'bong': {
            'required': {'covariance_type'},
            'unused': {'learning_rate', 'num_iter', 'batch_size'}
        },
    }
    method_spec = method_params.get(method, {})
    required_params = method_spec.get('required', set())
    unused_params = method_spec.get('unused', set())

    # Handle special cases    
    if cleaned_config['algorithm'].get('linplugin', False) and cleaned_config['algorithm'].get('num_samples', 1) != 1:
        # For linplugin, num_samples must be 1
        print(f"Warning: linplugin does not require sampling, setting num_samples to 1")
        cleaned_config['algorithm']['num_samples'] = 1

    if method in ['bbb', 'bog', 'blr'] and cleaned_config['algorithm']['covariance_type'] == 'full':
        # We only support diagonal covariance for these methods due to computational complexity
        print(f"Warning: {method} only supports diagonal covariance, setting covariance_type to diag")
        cleaned_config['algorithm']['covariance_type'] = 'diag'
    
    if model_type == 'resnet' and cleaned_config['algorithm'].get('covariance_type', None) == 'full':
        # Full covariance methods will run out of memory for ResNet model
        print(f"Warning: Full covariance methods will run out of memory for ResNet model, setting covariance_type to diag")
        cleaned_config['algorithm']['covariance_type'] = 'diag'

    if cleaned_config['algorithm'].get('covariance_type', None) == 'dlr':
        if cleaned_config['algorithm'].get('rank', None) is None:
            print(f"Missing required rank parameter for DLR covariance type, setting rank to 10")
            cleaned_config['algorithm']['rank'] = 10
    else:
        cleaned_config['algorithm'].pop('rank', None)

    if not(cleaned_config['algorithm'].get('linplugin', True) or cleaned_config['algorithm'].get('empirical_fisher', True)):
        # We don't allow MC-based methods gradients without empirical fisher due to computational complexity
        print(f"Warning: Running without linplugin requires empirical fisher, setting empirical_fisher to true")
        cleaned_config['algorithm']['empirical_fisher'] = True

    # Remove all unused parameters
    for param in unused_params:
        if param in cleaned_config['algorithm']:
            cleaned_config['algorithm'].pop(param)

    # Check that all required parameters are present
    for param in required_params:
        if param not in cleaned_config['algorithm']:
            raise ValueError(f"Missing required parameter: {param}")

    return cleaned_config

def create_model(config: dict[str, any], key: Array) -> Detector:
    """Create a DeepSIC model based on configuration."""
    model_config = config['model']
    channel_config = config['channel']
    model_type = model_config['type'].lower()

    model_class = DeepSIC if model_type == 'deepsic' else ResNetDetector
    model = model_class(
        key=key,
        symbol_bits=int(math.log2(len(MODULATIONS[channel_config['modulation']]))),
        num_users=channel_config['num_users'],
        num_antennas=channel_config['num_antennas'],
        num_layers=model_config['num_layers'],
        hidden_dim=model_config['hidden_dim']
    )
    return model

def create_channel(config: dict[str, any]) -> UplinkMimoChannel:
    """Create a channel based on configuration."""
    channel_config = config['channel']

    channel = UplinkMimoChannel(
        path=channel_config['channel_path'],
        modulation_type=channel_config['modulation'],
        num_users=channel_config['num_users'],
        num_antennas=channel_config['num_antennas'],
        apply_non_linearity=not channel_config['linear_channel']
    )
    return channel

def create_online_learning_function(config: dict[str, any], model: Detector) -> callable:
    """Create an online learning function based on algorithm configuration."""
    model_config = config['model']
    algo_config = config['algorithm']
    model_type = model_config['type'].lower()
    method = METHOD_MAP[algo_config['method'].lower()]

    if method == TrainingMethod.GD:
        init_state, extract_params, step_fn = build_gd_step_fn(
            apply_fn=model.apply_fn,
            loss_fn=optax.sigmoid_binary_cross_entropy,
            dynamics_decay=algo_config['dynamics_decay'],
            num_iter=algo_config['num_iter'],
            learning_rate=algo_config['learning_rate'],
        )
        return init_state, extract_params, step_fn

    elif method == TrainingMethod.SGD:
        init_state, extract_params, train_fn = build_sgd_train_fn(
            loss_fn=optax.sigmoid_binary_cross_entropy,
            dynamics_decay=algo_config['dynamics_decay'],
            num_epochs=algo_config['num_iter'],
            batch_size=algo_config['batch_size'],
            shuffle=True,
            optimizer=optax.adam,
            learning_rate=algo_config['learning_rate']
        )
        return init_state, extract_params, train_fn

    else:
        cov_type = COV_TYPE_MAP[algo_config['covariance_type'].lower()]
        obs_cov = algo_config['obs_cov_scale'] * jnp.eye(model.symbol_bits if model_type == 'deepsic' else model.output_size)
        if cov_type == CovarianceType.FULL:
            process_noise = jnp.eye(model.params.shape[-1])
        elif cov_type == CovarianceType.DLR:
            process_noise = 1
        elif cov_type == CovarianceType.DG:
            process_noise = jnp.ones(model.params.shape[-1])
        process_noise = algo_config['process_noise'] * process_noise

        init_state, extract_params, step_fn = step_fn_builder(
            method=method,
            apply_fn=model.apply_fn,
            obs_cov=obs_cov,
            covariance_type=cov_type,
            rank=algo_config.get('rank', 10),
            init_cov_scale=model_config['init_param_cov'],
            linplugin=algo_config['linplugin'],
            reparameterized=algo_config['reparameterized'],
            dynamics_decay=algo_config['dynamics_decay'],
            process_noise=process_noise,
            log_likelihood=None if algo_config['linplugin'] else lambda mean, cov, y: -optax.sigmoid_binary_cross_entropy(mean, y),
            num_samples=algo_config['num_samples'],
            empirical_fisher=algo_config['empirical_fisher'],
            learning_rate=algo_config.get('learning_rate', 0.1),
            num_iter=algo_config.get('num_iter', 1)
        )
        return init_state, extract_params, step_fn

def test_model(model: Detector, test_rx: jnp.ndarray, test_labels: jnp.ndarray) -> float:
    """Test model and return bit error rate."""
    predictions = model.soft_decode(test_rx)
    predicted_labels = (predictions > 0.5).astype(jnp.float32)
    accuracy = (predicted_labels == test_labels).mean()
    return float(1 - accuracy)

def measure_runtimes(
    model,
    state_init_fn: callable,
    extract_params: callable,
    online_learning_fn: callable,
    channel: UplinkMimoChannel,
    config: dict[str, any]
    ) -> tuple[float, float]:

    rx, labels = prepare_single_batch(channel=channel, num_samples=2048, frame_idx=0, snr=0, key=jr.PRNGKey(0))
    # First call contains compilation time
    if config['algorithm']['method'] == 'sgd':
        model.classic_fit(
                rx=rx,
                labels=labels,
                state_init_fn=state_init_fn,
                extract_params=extract_params,
                train_block_fn=online_learning_fn,
            )
        model.params.block_until_ready()
    else:
        model.streaming_fit(
            rx=rx,
            labels=labels,
            state_init_fn=state_init_fn,
            extract_params=extract_params,
            step_fn=online_learning_fn,
        )
        model.params.block_until_ready()
    result = model.soft_decode(rx)
    result.block_until_ready()

    start_time = time.time()
    if config['algorithm']['method'] == 'sgd':
        model.classic_fit(
                rx=rx,
                labels=labels,
                state_init_fn=state_init_fn,
                extract_params=extract_params,
                train_block_fn=online_learning_fn,
            )
        model.params.block_until_ready()
    else:
        model.streaming_fit(
            rx=rx,
            labels=labels,
            state_init_fn=state_init_fn,
            extract_params=extract_params,
            step_fn=online_learning_fn,
        )
        model.params.block_until_ready()
    training_time = (time.time() - start_time) / 2048
    start_time = time.time()
    model.soft_decode(rx).block_until_ready()
    inference_time = (time.time() - start_time) / 2048

    return training_time, inference_time

def run_experiment(config: dict[str, any]) -> tuple[dict[str, any], Detector]:
    """Run a single experiment based on configuration and return results and trained model."""
    key = jr.PRNGKey(config['experiment']['seed'])
    model_key, data_key = jr.split(key)
    train_key, test_key = jr.split(data_key)
    sync_key, track_key = jr.split(train_key)

    # Initialize channel, model, and step function
    channel = create_channel(config)
    model = create_model(config, model_key)
    state_init_fn, extract_params, online_learning_fn = create_online_learning_function(config, model)

    # Prepare data
    sync_frames = config['experiment']['sync_frames']
    track_frames = config['experiment']['track_frames']
    alloc_windows = int(config['experiment'].get('alloc_windows', 1))
    if alloc_windows < 1:
        alloc_windows = 1
    total_frames_per_window = sync_frames + track_frames

    if not total_frames_per_window > 0:
        raise ValueError("Total frames must be greater than 0")

    # Measure runtimes
    training_time, inference_time = measure_runtimes(model, state_init_fn, extract_params, online_learning_fn, channel, config)

    ber_array = []
    last_sync_end_index = None
    # Repeat cycles of sync + track, resetting the model each user allocation window
    for window_idx in tqdm(range(alloc_windows), total=alloc_windows, leave=False, desc='Allocation windows'):
        # Reset model parameters at the beginning of each user allocation window
        model = create_model(config, model_key)

        window_start = window_idx * (sync_frames + track_frames)

        test_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples=config['experiment']['test_dim'],
        num_frames=total_frames_per_window,
        snr=config['channel']['snr'],
        key=test_key,
        start_frame=window_start
        )
        test_dataloader_iterator = iter(test_dataloader)

        # Sync phase for this allocation window
        if sync_frames > 0:
            sync_dataloader = prepare_experiment_data(
                channel=channel,
                num_samples=config['experiment']['symbols_per_frame'],
                num_frames=sync_frames,
                snr=config['channel']['snr'],
                key=sync_key,
                start_frame=window_start
            )
            for train_rx, train_labels in tqdm(sync_dataloader, total=sync_frames, leave=False, desc='Sync frames'):
                if config['algorithm']['method'] == 'sgd':
                    model.classic_fit(
                        rx=train_rx,
                        labels=train_labels,
                        state_init_fn=state_init_fn,
                        extract_params=extract_params,
                        train_block_fn=online_learning_fn,
                    )
                else:
                    model.streaming_fit(
                        rx=train_rx,
                        labels=train_labels,
                        state_init_fn=state_init_fn,
                        extract_params=extract_params,
                        step_fn=online_learning_fn,
                        save_history=False,
                    )
                test_rx, test_labels = next(test_dataloader_iterator)
                ber = test_model(model, test_rx, test_labels)
                ber_array.append(ber)
            last_sync_end_index = len(ber_array) - 1

        # Track phase for this allocation window
        if track_frames > 0:
            track_dataloader = prepare_experiment_data(
                channel=channel,
                num_samples=config['experiment']['pilot_per_frame'],
                num_frames=track_frames,
                snr=config['channel']['snr'],
                key=track_key,
                start_frame=window_start + sync_frames
            )
            for train_rx, train_labels in tqdm(track_dataloader, total=track_frames, leave=False, desc='Track frames'):
                if config['algorithm']['method'] == 'sgd':
                    model.classic_fit(
                        rx=train_rx,
                        labels=train_labels,
                        state_init_fn=state_init_fn,
                        extract_params=extract_params,
                        train_block_fn=online_learning_fn,
                    )
                else:
                    model.streaming_fit(
                        rx=train_rx,
                        labels=train_labels,
                        state_init_fn=state_init_fn,
                        extract_params=extract_params,
                        step_fn=online_learning_fn,
                        save_history=False,
                    )
                test_rx, test_labels = next(test_dataloader_iterator)
                ber = test_model(model, test_rx, test_labels)
                ber_array.append(ber)

    # Compile results
    jnp_ber_array = jnp.array(ber_array)
    # Compute metrics over all track frames across allocation windows
    if track_frames > 0:
        track_segments = []
        for window_idx in range(alloc_windows):
            start_idx = window_idx * (sync_frames + track_frames) + sync_frames
            end_idx = start_idx + track_frames
            track_segments.append(jnp_ber_array[start_idx:end_idx])
        track_concat = jnp.concatenate(track_segments) if track_segments else jnp.array([])
    else:
        track_concat = jnp.array([])

    results = {
        'ber': ber_array,
        'final_sync_ber': float(jnp_ber_array[last_sync_end_index]) if last_sync_end_index is not None else None,
        'avg_track_ber': float(jnp.mean(track_concat)) if track_concat.size > 0 else None,
        'std_track_ber': float(jnp.std(track_concat)) if track_concat.size > 0 else None,
        'p95_track_ber': float(jnp.percentile(track_concat, 95)) if track_concat.size > 0 else None,
        'p99_track_ber': float(jnp.percentile(track_concat, 99)) if track_concat.size > 0 else None,
        'training_time_per_sample': training_time,
        'inference_time_per_sample': inference_time,
    }
    jax.clear_caches()
    return results, model

def save_results(results: dict[str, any], model: Detector, config: dict[str, any], base_output_dir: str) -> str:
    """Save experiment results and model state to hash-based folder structure."""
    config_hash = generate_config_hash(config)
    output_dir = os.path.join(base_output_dir, config_hash)
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration file
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Save model state (parameters)
    model.save(path=os.path.join(output_dir, 'model_state.pkl'))

    # Clean results
    results_clean = {}
    for key, value in results.items():
        if key == 'config':
            continue
        elif isinstance(value, jnp.ndarray):
            results_clean[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], jnp.ndarray):
            results_clean[key] = [v.tolist() for v in value]
        else:
            results_clean[key] = value
    results_clean['config_hash'] = config_hash

    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results_clean, f, indent=2)

def load_experiment_by_hash(config_hash: str, base_results_dir: str = 'adr/experiments/results') -> dict[str, any]:
    """Load experiment results and config by hash."""
    experiment_dir = os.path.join(base_results_dir, config_hash)

    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"No experiment found with hash: {config_hash}")

    # Load results
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Load config
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load model state if needed
    model_path = os.path.join(experiment_dir, 'model_state.pkl')
    model_state = None
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_state = pickle.load(f)

    return {
        'results': results,
        'config': config,
        'model_state': model_state,
    }

def main():
    """Main function to run experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment from JSON config')
    parser.add_argument('--config_path', type=str, help='Path to experiment config JSON file', default='adr/experiments/single_config.json')
    parser.add_argument('--output_dir', type=str, help='Base output directory for results', default='adr/experiments/results')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    validate_config(config)
    config = clean_config(config)
    config_hash = generate_config_hash(config)

    # Check if experiment already exists
    experiment_dir = os.path.join(args.output_dir, config_hash)
    if os.path.exists(experiment_dir):
        return

    # Run experiment
    results, model = run_experiment(config)
    save_results(results, model, config, args.output_dir)

if __name__ == "__main__":
    main()
