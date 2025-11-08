import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
import optax
import math
from tqdm import tqdm
from adr import ResNetDetector, CovarianceType, TrainingMethod, step_fn_builder, build_sgd_train_fn
from adr import bit_array_to_index, index_to_bit_array

# Plotting settings
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Save directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QPSK_CONSTELATION = np.array([
    [1, 1],      # 00
    [-1, 1],     # 01
    [1, -1],     # 10
    [-1, -1]     # 11
]) / np.sqrt(2)

class LinearPhaseChannel:
    """
    Linear phase channel: h(t) = exp(j * alpha * t)

    Args:
        alpha (float): Phase rotation speed (radians per time step).
        Es (float, optional): Symbol energy. Defaults to 1.0.
        noise_var (float, optional): Noise variance (N0/2). Defaults to 0.0625.
    """

    def __init__(self, alpha: float, Es: float = 1.0, noise_var: float = 0.0625):
        self.alpha = alpha
        self.Es = Es
        self.noise_var = noise_var
        self.time = 0

        # QPSK constellation points (π/4-QPSK)
        self.constellation = np.sqrt(Es) * QPSK_CONSTELATION

        # Store constellation history for visualization
        self.constellation_history = []

    def get_channel(self, t: int = None) -> complex:
        """Get channel value at time t"""
        if t is None:
            t = self.time
        return np.exp(1j * self.alpha * t)

    def get_rotated_constellation(self, t: int = None) -> np.ndarray:
        """Get constellation points rotated by the channel at time t"""
        h = self.get_channel(t)
        h_real = np.real(h)
        h_imag = np.imag(h)
        rotation_matrix = np.array([[h_real, -h_imag], [h_imag, h_real]])
        rotated_const = rotation_matrix @ self.constellation.T
        return rotated_const.T

    def generate_samples(self, n_samples: int, randomize_order: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Generate n_samples from the current rotated constellation."""
        rotated_const = self.get_rotated_constellation()
        if randomize_order:
            labels = np.random.choice(4, size=n_samples)
        else:
            labels = np.arange(n_samples) % 4
        samples = rotated_const[labels] + np.sqrt(self.noise_var) * np.random.randn(n_samples, 2)
        self.constellation_history.append(rotated_const.copy())

        return samples, labels

    def advance_time(self):
        """Advance time by one step"""
        self.time += 1

    def reset(self):
        """Reset time and clear history"""
        self.time = 0
        self.constellation_history = []

    def get_theoretical_error_rate(self) -> float:
        """Calculate theoretical MAP error rate"""
        SNR = 2 * self.Es / (2 * self.noise_var)  # SNR = 2*Es/N0
        Q_val = 0.5 * (1 - math.erf(np.sqrt(SNR/2) / np.sqrt(2)))
        Pe = 1 - (1 - Q_val)**2
        return Pe


class NLMSFilter:
    """
    Normalized Least Mean Squares (NLMS) filter for tracking linear phase channel.

    Args:
        mu (float, optional): Step size (learning rate). Default is 0.1.
        epsilon (float, optional): Small constant to prevent division by zero. Default is 1e-6.
    """

    def __init__(self, mu: float = 0.1, epsilon: float = 1e-6):
        self.mu = mu
        self.epsilon = epsilon

        # Initialize channel estimate as complex number
        self.channel_estimate = 0.5 + 0.5j

        # History for tracking
        self.channel_history = []
        self.error_history = []

    def update(self, received_signal: np.ndarray, transmitted_symbol: np.ndarray):
        """Update the channel estimate using NLMS algorithm."""
        received_complex = received_signal[0] + 1j * received_signal[1]
        transmitted_complex = transmitted_symbol[0] + 1j * transmitted_symbol[1]

        # Compute error
        predicted_received = self.channel_estimate * transmitted_complex
        error = received_complex - predicted_received

        # Normalize step size
        input_power = abs(transmitted_complex)**2 + self.epsilon
        normalized_step = self.mu / input_power

        # Update channel estimate
        self.channel_estimate += normalized_step * error * np.conj(transmitted_complex)

        # Store history
        self.channel_history.append(self.channel_estimate)
        self.error_history.append(error)

    def get_channel_estimate(self) -> complex:
        """Get current channel estimate as complex number."""
        return self.channel_estimate

    def reset(self):
        """Reset filter weights and history."""
        self.channel_estimate = 0.5 + 0.5j
        self.channel_history = []
        self.error_history = []


class NLMSDetector:
    """
    NLMS-based detector for QPSK symbols with linear phase channel tracking.

    This detector uses NLMS filter to track the channel and then performs maximum likelihood detection on the equalized signal.

    Args:
        Es (float): Symbol energy.
        mu (float, optional): Step size (learning rate). Default is 0.1.
        epsilon (float, optional): Small constant to prevent division by zero. Default is 1e-6.
    """

    def __init__(self, Es: float, mu: float = 0.1, epsilon: float = 1e-6):
        self.nlms_filter = NLMSFilter(mu=mu, epsilon=epsilon)
        self.constellation = np.sqrt(Es) * QPSK_CONSTELATION

    def detect_symbols(self, received_signals: np.ndarray, training_symbols: np.ndarray = None) -> np.ndarray:
        """Detect multiple QPSK symbols in batch using vectorized operations."""
        # Update filter if training symbols are provided
        if training_symbols is not None:
            for signal, training in zip(received_signals, training_symbols):
                self.nlms_filter.update(signal, training)

        # Get channel estimate and convert received signals to complex
        channel_estimate = self.nlms_filter.get_channel_estimate()
        received_complex = received_signals[:, 0] + 1j * received_signals[:, 1]

        # Equalize
        if abs(channel_estimate) > 1e-6:
            equalized_complex = received_complex / channel_estimate
        else:
            equalized_complex = received_complex

        equalized_signals = np.stack([np.real(equalized_complex), np.imag(equalized_complex)], axis=1)
        equalized_signals = equalized_signals[:, np.newaxis, :]

        # Demap
        distances = np.linalg.norm(self.constellation[np.newaxis, :, :] - equalized_signals, axis=2)
        detected_symbols = np.argmin(distances, axis=1)

        return detected_symbols


class MAPDetector:
    """
    MAP detector for QPSK symbols.

    Args:
        Es (float): Symbol energy.
    """

    def __init__(self, Es: float):
        self.constellation = np.sqrt(Es) * QPSK_CONSTELATION

    def detect_symbols(self, received_signals: np.ndarray, channel: complex) -> np.ndarray:
        """Detect multiple QPSK symbols in batch using vectorized operations."""
        # Convert received signals to complex numbers
        received_complex = received_signals[:, 0] + 1j * received_signals[:, 1]
        
        # Equalize
        equalized_complex = received_complex / channel
        equalized_signals = np.stack([np.real(equalized_complex), np.imag(equalized_complex)], axis=1)
        equalized_signals = equalized_signals[:, np.newaxis, :]
        
        # Demap
        distances = np.linalg.norm(self.constellation[np.newaxis, :, :] - equalized_signals, axis=2)
        detected_symbols = np.argmin(distances, axis=1)
        
        return detected_symbols

def visualize_constellation_trajectory(Es: float, noise_var: float, alpha: float, n_steps: int):
    """Visualize how constellation points move over time."""
    channel = LinearPhaseChannel(alpha=alpha, Es=Es, noise_var=noise_var)

    for i in range(n_steps):
        channel.generate_samples(16, randomize_order=True)
        channel.advance_time()

    constellation_history = np.array(channel.constellation_history)

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'red']
    labels = ['00', '01', '10', '11']

    for i in range(4):
        plt.plot(constellation_history[:, i, 0], constellation_history[:, i, 1], 
                 color=colors[i], alpha=0.7, linewidth=2, label=f'Symbol {labels[i]}')
        plt.scatter(constellation_history[0, i, 0], constellation_history[0, i, 1], 
                    c=colors[i], s=100, marker='o', edgecolors='black')
        plt.scatter(constellation_history[-1, i, 0], constellation_history[-1, i, 1], 
                    c=colors[i], s=100, marker='s', edgecolors='black')

    plt.xlabel('In-phase component', fontsize=14)
    plt.ylabel('Quadrature component', fontsize=14)
    plt.title(f'Constellation Trajectories (α={alpha:.4f})', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axis('equal')
    plt.savefig(os.path.join(BASE_DIR, 'constellation_trajectories.pdf'), bbox_inches='tight')
    plt.close()

def create_single_user_model(hidden_dim: int):
    """Create DeepSIC models for single user scenario"""

    num_users = 1
    num_antennas = 1
    symbol_bits = 2
    num_layers = 0
    key = jr.PRNGKey(42)

    deepsic_model = ResNetDetector(
        key=key,
        symbol_bits=symbol_bits,
        num_users=num_users,
        num_antennas=num_antennas,
        num_layers=num_layers,
        hidden_dim=hidden_dim
    )

    return deepsic_model

def run_experiments(
    Es: float,
    noise_var: float,
    alpha: float,
    n_steps: int,
    pilots_per_step: int,
    test_dim: int,
    hidden_dim: int,
    nlms_step_size: float,
    sgd_num_epochs: int,
    sgd_batch_size: int,
    sgd_learning_rate: float,
    ekf_obs_cov: float,
    ekf_dynamics_decay: float,
    ekf_process_noise: float
    ):
    """Run experiments.

    Args:
        Es (float): Symbol energy.
        noise_var (float): Noise variance.
        alpha (float): Phase rotation speed.
        n_steps (int): Number of steps.
        pilots_per_step (int): Number of pilots per step.
        test_dim (int): Number of test samples.
        hidden_dim (int): Number of hidden dimensions.
        nlms_step_size (float): NLMS step size.
        sgd_num_epochs (int): Number of SGD epochs.
        sgd_batch_size (int): SGD batch size.
        sgd_learning_rate (float): SGD learning rate.
        ekf_obs_cov (float): EKF observation covariance.
        ekf_dynamics_decay (float): EKF dynamics decay.
        ekf_process_noise (float): EKF process noise.
        
    """
    channel = LinearPhaseChannel(alpha=alpha, Es=Es, noise_var=noise_var)

    # Create detectors
    map_error_rate = channel.get_theoretical_error_rate()
    map_detector = MAPDetector(Es=Es)
    nlms_detector = NLMSDetector(Es=Es, mu=nlms_step_size, epsilon=1e-6)
    bnn_detector = create_single_user_model(hidden_dim=hidden_dim)
    ekf_state_init, ekf_extract_params, ekf_step_fn = step_fn_builder(
        method=TrainingMethod.BONG,
        apply_fn=bnn_detector.apply_fn,
        obs_cov=ekf_obs_cov * jnp.eye(2),
        covariance_type=CovarianceType.FULL,
        linplugin=True,
        reparameterized=False,
        log_likelihood = None,
        dynamics_decay=ekf_dynamics_decay,
        process_noise=ekf_process_noise * jnp.eye(bnn_detector.params.shape[-1]),
        empirical_fisher=False,
        num_iter=1
        )
    dnn_detector = create_single_user_model(hidden_dim=hidden_dim)
    sgd_state_init, sgd_extract_params, sgd_train_fn = build_sgd_train_fn(
        loss_fn=optax.sigmoid_binary_cross_entropy,
        dynamics_decay=1.0,
        num_epochs=sgd_num_epochs,
        batch_size=sgd_batch_size,
        optimizer=optax.adam,
        learning_rate=sgd_learning_rate
        )


    # Run training
    methods = ["MAP", "NLMS", "CM-EKF", "SGD"]
    error_rates = [[] for _ in methods]

    for _ in tqdm(range(n_steps)):
        samples, labels = channel.generate_samples(pilots_per_step, randomize_order=True)
        test_samples, test_labels = channel.generate_samples(test_dim, randomize_order=False)
        channel.advance_time()

        for i, method in enumerate(methods):
            if method == "MAP":
                error_rates[i].append(map_error_rate)
            elif method == "NLMS":
                nlms_detector.detect_symbols(samples, channel.constellation[labels])
                decoded_symbols = nlms_detector.detect_symbols(test_samples)
                error_rates[i].append(np.mean(decoded_symbols != test_labels))
            elif method == "CM-EKF":
                bitwise_labels = jnp.expand_dims(index_to_bit_array(labels, 2), 1)
                bnn_detector.streaming_fit(jnp.array(samples), jnp.array(bitwise_labels), ekf_state_init, ekf_extract_params, ekf_step_fn)
                soft_decoded_bits = bnn_detector.soft_decode(test_samples).reshape(test_dim, 2)
                decoded_bits = (soft_decoded_bits > 0.5).astype(jnp.int32)
                decoded_symbols = bit_array_to_index(decoded_bits)
                error_rates[i].append(np.mean(decoded_symbols != test_labels))
            elif method.startswith("SGD"):
                bitwise_labels = jnp.expand_dims(index_to_bit_array(labels, 2), 1)
                dnn_detector.classic_fit(jnp.array(samples), jnp.array(bitwise_labels), sgd_state_init, sgd_extract_params, sgd_train_fn)
                soft_decoded_bits = dnn_detector.soft_decode(test_samples).reshape(test_dim, 2)
                decoded_bits = (soft_decoded_bits > 0.5).astype(jnp.int32)
                decoded_symbols = bit_array_to_index(decoded_bits)
                error_rates[i].append(np.mean(decoded_symbols != test_labels))

    # Create the main plot
    _, ax = plt.subplots()

    ax.plot(error_rates[3], label=f"SGD-{sgd_num_epochs}-{sgd_batch_size}", marker="X", markevery=10, color="#ff7f0e", alpha=0.8)
    ax.plot(error_rates[2], label="CM-EKF", marker="o", markevery=10, color="#2ca02c", alpha=0.8)
    ax.plot(error_rates[1], label="NLMS", marker="p", markevery=10, color="#e377c2", alpha=0.8)
    ax.plot(error_rates[0], label="MAP", linestyle="-.", color="black", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Channel Snapshot", fontsize=18)
    ax.set_ylabel("Symbol Error Rate", fontsize=18)
    ax.tick_params(axis='x', labelsize=14) 
    ax.tick_params(axis='y', labelsize=14) 

    # Create the zoomed plot
    axins = ax.inset_axes([0.3, 0.6, 0.3, 0.3])
    axins.plot(error_rates[3], marker="X", markevery=10, color="#ff7f0e", alpha=0.8)
    axins.plot(error_rates[2], marker="o", markevery=10, color="#2ca02c", alpha=0.8)
    axins.plot(error_rates[1], marker="p", markevery=10, color="#e377c2", alpha=0.8)
    axins.plot(error_rates[0], linestyle="-.", color="black", alpha=0.8)
    axins.set_xlim(110, 190)
    axins.set_ylim(4e-3, 7e-3)
    axins.set_yscale("log")
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.grid()
    ax.legend(loc='upper right', fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(BASE_DIR, "error_rates.pdf"), bbox_inches='tight')
    plt.close()

    detectors = {"MAP": map_detector, "NLMS": nlms_detector, "CM-EKF": bnn_detector, f"SGD-{sgd_num_epochs}-{sgd_batch_size}": dnn_detector}

    return channel, detectors

def plot_decision_zones(channel: LinearPhaseChannel, detectors: dict):
    """Visualize decision zones of trained decoders using a grid-based approach."""

    # Build grid
    samples, _ = channel.generate_samples(1000)
    x_min, x_max = samples[:, 0].min() - 0.5, samples[:, 0].max() + 0.5
    y_min, y_max = samples[:, 1].min() - 0.5, samples[:, 1].max() + 0.5
    grid_size = max(abs(x_min), abs(y_min), abs(x_max), abs(y_max))
    grid_size = min(grid_size, 3.0)
    grid_resolution = 0.01
    xx, yy = np.meshgrid(
        np.arange(-grid_size, grid_size, grid_resolution),
        np.arange(-grid_size, grid_size, grid_resolution)
    )

    # Create subplots for each detector
    _, axes = plt.subplots(1, len(detectors), figsize=(5 * len(detectors), 5))
    axes = axes.flatten()
    contour_colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e", "#2ca02c", "#2ca02c", "#d62728", "#d62728"]
    scatter_colors = ["blue", "orange", "green", "red"]
    sample_points, sample_labels = channel.generate_samples(1000)

    for idx, (method, detector) in enumerate(detectors.items()):
        ax = axes[idx]
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        if method == "MAP":
            grid_predictions = detector.detect_symbols(grid_points, channel.get_channel())
        elif method == "NLMS":
            grid_predictions = detector.detect_symbols(grid_points)
        else:
            soft_decoded_bits = detector.soft_decode(grid_points).reshape(-1, 2)
            decoded_bits = (soft_decoded_bits > 0.5).astype(jnp.int32)
            grid_predictions = bit_array_to_index(decoded_bits)
        grid_predictions = grid_predictions.reshape(xx.shape)

        # Plot decision zones
        ax.contourf(xx, yy, grid_predictions, alpha=0.7, colors=contour_colors)
        ax.contour(xx, yy, grid_predictions, alpha=0.7, colors='black', linewidths=1.5)

        # Plot some sample points
        ax.scatter(sample_points[:, 0], sample_points[:, 1], 
                    c=[scatter_colors[label] for label in sample_labels], 
                    alpha=0.7, s=20, edgecolors='k', linewidth=0.5)

        ax.set_title(f'{method}', fontsize=14, fontweight='bold')
        ax.set_xlabel('In-phase component', fontsize=14)
        ax.set_ylabel('Quadrature component', fontsize=14)
        ax.set_xlim([-grid_size, grid_size])
        ax.set_ylim([-grid_size, grid_size])

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"decision_zones.png"), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":

    # Set random seeds for reproducibility
    np.random.seed(0)

    print("Linear Phase Channel Tracking Comparison")
    print("========================================")

    Es = 1.0
    noise_var = 0.0625
    alpha = math.pi / 2000
    n_steps = 500
    pilots_per_step = 16
    test_dim = 100000
    hidden_dim = 10
    nlms_step_size = 0.01
    sgd_num_epochs = 8
    sgd_batch_size = 4
    sgd_learning_rate = 1e-3
    ekf_obs_cov = 0.1
    ekf_dynamics_decay = 0.999
    ekf_process_noise = 5e-5

    print("Experiment parameters:")
    print(f"  Es: {Es}")
    print(f"  Noise variance: {noise_var}")
    print(f"  Alpha: {alpha}")
    print(f"  Number of steps: {n_steps}")
    print(f"  Symbols per step: {pilots_per_step}")
    
    print("\n1. Visualizing constellation trajectories...")
    visualize_constellation_trajectory(Es=Es, noise_var=noise_var, alpha=alpha, n_steps=n_steps)
    
    print("\n2. Running comparison...")
    channel, detectors = run_experiments(
        Es=Es,
        noise_var=noise_var,
        alpha=alpha,
        n_steps=n_steps,
        pilots_per_step=pilots_per_step,
        test_dim=test_dim,
        hidden_dim=hidden_dim,
        nlms_step_size=nlms_step_size,
        sgd_num_epochs=sgd_num_epochs,
        sgd_batch_size=sgd_batch_size,
        sgd_learning_rate=sgd_learning_rate,
        ekf_obs_cov=ekf_obs_cov,
        ekf_dynamics_decay=ekf_dynamics_decay,
        ekf_process_noise=ekf_process_noise
        )
    plot_decision_zones(channel, detectors)

    print("\nDone!")
