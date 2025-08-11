from typing import Any
import hashlib
import json
import jax
from jax import Array
import jax.numpy as jnp
from jax_dataloader import ArrayDataset, DataLoader
from adr import UplinkMimoChannel

#########################
#   Experiment utils    #
#########################

def load_config(config_path: str) -> dict[str, Any]:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def generate_config_hash(config: dict[str, Any]) -> str:
    """Generate a deterministic hash from the experiment configuration."""

    config_for_hash = config.copy()
    
    # Convert to JSON string with sorted keys for deterministic hashing and remove spaces
    config_str = json.dumps(config_for_hash, sort_keys=True, separators=(',', ':')).replace(' ', '')
    
    # Generate hash
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:12]  # Use first 12 characters for readability

#########################
#      Data utils       #
#########################

def prepare_single_batch(channel: UplinkMimoChannel, num_samples: int, frame_idx: int, snr: int, key: Array) -> tuple[Array, Array]:
    """Prepares online training experiment data for a single time frame.

    Args:
        channel (UplinkMimoChannel): Channel used to generate data.
        num_samples (int): Number of samples per time frame.
        frame_idx (int): Index of the time frame.
        snr (int): Signal-to-noise ratio.
        key (Array): Random key for noise generation.
    """
    num_users = channel.num_users
    labels_key, transmit_key = jax.random.split(key)
    label_bits = jax.random.randint(
        labels_key,
        shape=(num_samples, num_users, int(jnp.log2(channel.constellation_points.shape[0]))),
        minval=0,
        maxval=2,
        dtype=jnp.int32
    )
    labels = label_bits[..., 0]
    for i in range(1, label_bits.shape[-1]):
        labels = labels * 2 + label_bits[..., i]
    rx = channel.transmit(key=transmit_key, s=labels, snr=snr, frame_idx=frame_idx)
    if channel.modulation_type != 'BPSK':
        rx = jnp.stack([jnp.real(rx), jnp.imag(rx)], axis=-1)
    rx = rx.reshape(num_samples, -1)

    return rx, label_bits

def prepare_experiment_data(channel: UplinkMimoChannel, num_samples: int, num_frames: int, snr: int, key: Array, start_frame: int = 0) -> DataLoader:
    """Prepares data for online training experiments.

    Args:
        channel (UplinkMimoChannel): Channel used to generate data.
        num_samples (int): Number of samples per time frame.
        num_frames (int): Number of time frames.
        snr (int): Signal-to-noise ratio.
        key (Array): Random key for noise generation.
        start_frame (int): Starting frame index for data generation. Defaults is 0.
    """
    subkeys = jax.random.split(key, num_frames)
    label_blocks = jnp.zeros((0, 0))
    receive_blocks = jnp.zeros((0, 0))
    for frame_idx in range(start_frame, start_frame + num_frames):
        rx, labels = prepare_single_batch(channel, num_samples, frame_idx, snr, subkeys[frame_idx] if key is not None else None)
        label_blocks = labels if frame_idx == start_frame else jnp.concatenate([label_blocks, labels], axis=0)
        receive_blocks = rx if frame_idx == start_frame else jnp.concatenate([receive_blocks, rx], axis=0)

    dataset = ArrayDataset(receive_blocks, label_blocks)
    dataloader = DataLoader(dataset, 'jax', batch_size=num_samples, shuffle=False)
    return dataloader
