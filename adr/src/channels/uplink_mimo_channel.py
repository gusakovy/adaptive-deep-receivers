import os
import scipy.io
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from adr.src.channels.base import Channel
from adr.src.channels.modulations import MODULATIONS


class UplinkMimoChannel(Channel):
    """Memory-less time-varying uplink MIMO channel.

    Args:
        channel_dir (str): Path to the directory containing the channel matrices.
        file_prefix (str): Prefix of the channel matrix files, each corresponding to one configuration.
        frames_per_config (int): Number of time frames per configuration.
        modulation_type (str): Modulation type, e.g., "BPSK", "QPSK", "16-QAM".
        scaling_coefficient (float): Scaling coefficient for the channel matrix.
        num_users (int): Number of single-antenna users.
        num_antennas (int): Number of receive antennas.
        apply_non_linearity (bool, optional): Whether to apply non-linearity to the received signal. Defaults to False.
        keep_in_memory (bool, optional): Whether to keep the loaded channel matrices in memory. Defaults to False.
    """

    def __init__(
            self,
            channel_dir: str,
            file_prefix: str,
            frames_per_config: int,
            modulation_type: str,
            scaling_coefficient: float,
            num_users: int,
            num_antennas: int,
            apply_non_linearity: bool = False,
            keep_in_memory: bool = False
        ):
        if not os.path.exists(channel_dir):
            raise FileNotFoundError(f"Directory {channel_dir} does not exist.")
        self.channel_dir = channel_dir
        if not os.path.exists(os.path.join(self.channel_dir, f'{file_prefix}_1.mat')):
            raise FileNotFoundError(f"File {file_prefix}_1.mat does not exist in {self.channel_dir}.")
        self.file_prefix = file_prefix
        self.frames_per_config = frames_per_config
        self.num_frames = len([f for f in os.listdir(self.channel_dir) if f.startswith(self.file_prefix)]) * self.frames_per_config
        if modulation_type not in MODULATIONS.keys():
            raise ValueError(f"Modulation type must be one of {MODULATIONS.keys()}.")
        self.modulation_type = modulation_type
        self.constellation_points = MODULATIONS[self.modulation_type]
        self.scaling_coefficient = scaling_coefficient
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.apply_non_linearity = apply_non_linearity
        self.keep_in_memory = keep_in_memory
        self.h = [None for _ in range(self.num_frames)] if self.keep_in_memory else None

    def preload_channel_matrices(self, num_frames: int) -> Array:
        if self.keep_in_memory:
            for i in range(num_frames):
                self._compute_channel_signal_convolution(i)
        else:
            raise ValueError("Preloading not allowed, set keep_in_memory=True to allow preloading them.")

    def _calculate_channel(self, frame_idx: int) -> Array:
        """Calculate the channel information matrix."""
        if not self.h or self.h[frame_idx] is None:
            h = jnp.zeros((self.num_users, self.num_antennas), dtype=jnp.complex64)
            main_file_num = 1 + (frame_idx // self.frames_per_config)
            for i in range(0, self.num_users):
                path_to_mat = os.path.join(self.channel_dir, f'{self.file_prefix}_{main_file_num}.mat')
                h_user = self.scaling_coefficient * jnp.array(
                    scipy.io.loadmat(path_to_mat)['H'][i, :self.num_antennas, frame_idx % self.frames_per_config],
                    dtype=jnp.complex64
                )
                h_user = h_user / max(abs(h_user))
                h = h.at[i].set(h_user)
            if self.keep_in_memory:
                self.h[frame_idx] = h
        else:  # If the channel matrix is already calculated, use it
            h = self.h[frame_idx]
        return h

    @staticmethod
    @jax.jit
    def _compute_channel_signal_convolution(h: Array, tx: Array) -> Array:
        """Compute the convolution of a channel matrix and an array of constellation points."""
        conv = tx @ h
        return conv

    def transmit(self, key: Array, s: Array, snr: float, frame_idx: int = 0) -> Array:
        """Simulate transmission of symbols.

        Args:
            key (Array): Random key for noise generation.
            s (Array): Symbols to be transmitted.
            snr (float): Signal-to-noise ratio.
            frame_idx (int, optional): Time frame index. Defaults to 0.
        """
        h = self._calculate_channel(frame_idx)
        tx = self.constellation_points[s].reshape(-1, self.num_users)
        conv = UplinkMimoChannel._compute_channel_signal_convolution(h, tx)
        var = 10 ** (-0.1 * snr)
        if self.modulation_type == "BPSK":
            w = jnp.sqrt(var) * jr.normal(key, (tx.shape[0], self.num_antennas))
        else:
            subkeys = jr.split(key, 2)
            w_real = jnp.sqrt(var) / 2 * jr.normal(subkeys[0], (tx.shape[0], self.num_antennas))
            w_imag = jnp.sqrt(var) / 2 * jr.normal(subkeys[1], (tx.shape[0], self.num_antennas)) * 1j
            w = w_real + w_imag
        y = conv + w
        if self.apply_non_linearity:
            y = jnp.tanh(0.5 * y)
        return y
