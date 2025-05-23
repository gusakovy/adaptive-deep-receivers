from jax import Array
import jax.numpy as jnp
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM
from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_filter

def iterative_ekf(
        init_param_mean: Array,
        unravel_fn: callable,
        apply_fn: callable,
        inputs: Array,
        labels: Array,
        init_param_cov: float | Array,
        param_dynamics_function: callable,
        param_dynamics_cov: float | Array,
        obs_function: callable,
        obs_cov: Array,
        **kwargs
    ) -> tuple[Array, Array]:
    """Train a neural network using an (iterated) extended Kalman filter (EKF) algorithm.

    Args:
        init_param_mean (Array): Initial parameters.
        unravel_fn (callable): Parameter unravel function.
        apply_fn (callable): Model apply function.
        inputs (Array): Model input(s).
        labels (Array): Corresponding label(s).
        init_param_cov (float | Array): Initial parameter covariance.
        param_dynamics_function (callable): Parameter dynamics function.
        param_dynamics_cov (float | Array): Parameter dynamics covariance.
        obs_function (callable): Observation function.
        obs_cov (Array): Observation covariance.
    """
    init_param_cov = init_param_cov if isinstance(init_param_cov, Array) else init_param_cov * jnp.eye(len(init_param_mean))
    param_dynamics_cov = param_dynamics_cov if isinstance(param_dynamics_cov, Array) else param_dynamics_cov * jnp.eye(len(init_param_mean))

    ekf_params = ParamsNLGSSM(
        initial_mean=init_param_mean,
        initial_covariance=init_param_cov,
        dynamics_function=param_dynamics_function,
        dynamics_covariance=param_dynamics_cov,
        emission_function=lambda w, x: obs_function(apply_fn(unravel_fn(w), x)),
        emission_covariance=obs_cov
    )

    ekf_results = extended_kalman_filter(ekf_params, labels, inputs=inputs)
    new_param_mean = ekf_results.filtered_means[-1]
    new_param_cov = ekf_results.filtered_covariances[-1]
    return new_param_mean, new_param_cov
