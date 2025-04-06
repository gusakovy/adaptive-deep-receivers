from jax import Array
import jax.numpy as jnp
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM
from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_filter
from bong.src import bong, bog, bbb
from bong.util import run_rebayes_algorithm

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
        emission_covariance=obs_cov,
    )

    ekf_results = extended_kalman_filter(ekf_params, labels, inputs=inputs)
    new_param_mean = ekf_results.filtered_means[-1]
    new_param_cov = ekf_results.filtered_covariances[-1]
    return new_param_mean, new_param_cov

def fg_bong(
        init_param_mean: Array,
        unravel_fn: callable,
        apply_fn: callable,
        inputs: Array,
        labels: Array,
        key: Array,
        init_param_cov: float | Array,
        log_likelihood: callable,
        obs_function: callable,
        obs_cov: Array,
        dynamics_decay: float = 1.0,
        process_noise: float = 0.0,
        num_samples: int = 10,
        linplugin: bool = True,
        empirical_fisher: bool = False,
        **kwargs
    ) -> tuple[Array, Array]:
    """Train a neural network using the full-covariance Gaussian Bayesian online natural gradient (BONG) algorithm.

    Args:
        init_param_mean (Array): Initial parameters.
        unravel_fn (callable): Parameter unravel function.
        apply_fn (callable): Model apply function.
        inputs (Array): Model input(s).
        labels (Array): Corresponding label(s).
        key (Array): JAX PRNG Key.
        init_param_cov (float | Array): Initial parameter covariance.
        log_likelihood (callable): Log-likelihood function.
        obs_function (callable, optional): Observation function. Defaults to identity function.
        obs_cov (Array): Observation covariance.
        dynamics_decay (float, optional): Parameter dynamics decay. Defaults to 1.0.
        process_noise (float, optional): Parameter process noise. Defaults to 0.0.
        num_samples (int, optional): Number of samples to use for each update. Defaults to 10.
        linplugin (bool, optional): Whether to use the linearized plugin method. Defaults to True.
        empirical_fisher (bool, optional): Whether to use the empirical Fisher approximation to the Hessian matrix. Defaults to False.
    """

    init_param_cov = init_param_cov if isinstance(init_param_cov, Array) else init_param_cov * jnp.eye(len(init_param_mean))

    fg_bong = bong.fg_bong(
        init_mean=init_param_mean,
        init_cov=init_param_cov,
        log_likelihood=log_likelihood,
        emission_mean_function=lambda w, x: obs_function(apply_fn(unravel_fn(w), x)),
        emission_cov_function=lambda w, x: obs_cov,
        dynamics_decay=dynamics_decay,
        process_noise=process_noise,
        num_samples=num_samples,
        linplugin=linplugin,
        empirical_fisher=empirical_fisher
    )

    bong_result, _ = run_rebayes_algorithm(key, fg_bong, inputs, labels)
    return bong_result.mean, bong_result.cov

def fg_bog(
        init_param_mean: Array,
        unravel_fn: callable,
        apply_fn: callable,
        inputs: Array,
        labels: Array,
        key: Array,
        init_param_cov: float | Array,
        log_likelihood: callable,
        obs_function: callable,
        obs_cov: Array,
        dynamics_decay: float = 1.0,
        process_noise: float = 0.0,
        num_samples: int = 10,
        linplugin: bool = True,
        empirical_fisher: bool = False,
        learning_rate: float = 0.1,
        **kwargs
    ) -> tuple[Array, Array]:
    """Train a neural network using the full-covariance Gaussian Bayesian online gradient (BOG) algorithm.

    Args:
        init_param_mean (Array): Initial parameters.
        unravel_fn (callable): Parameter unravel function.
        apply_fn (callable): Model apply function.
        inputs (Array): Model input(s).
        labels (Array): Corresponding label(s).
        key (Array): JAX PRNG Key.
        init_param_cov (float | Array): Initial parameter covariance.
        log_likelihood (callable): Log-likelihood function.
        obs_function (callable, optional): Observation function. Defaults to identity function.
        obs_cov (Array): Observation covariance.
        dynamics_decay (float, optional): Parameter dynamics decay. Defaults to 1.0.
        process_noise (float, optional): Parameter process noise. Defaults to 0.0.
        num_samples (int, optional): Number of samples to use for each update. Defaults to 10.
        linplugin (bool, optional): Whether to use the linearized plugin method. Defaults to True.
        empirical_fisher (bool, optional): Whether to use the empirical Fisher approximation to the Hessian matrix. Defaults to False.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.1.
    """
    init_param_cov = init_param_cov if isinstance(init_param_cov, Array) else init_param_cov * jnp.eye(len(init_param_mean))

    fg_bog = bog.fg_bog(
        init_mean=init_param_mean,
        init_cov=init_param_cov,
        log_likelihood=log_likelihood,
        emission_mean_function=lambda w, x: obs_function(apply_fn(unravel_fn(w), x)),
        emission_cov_function=lambda w, x: obs_cov,
        dynamics_decay=dynamics_decay,
        process_noise=process_noise,
        num_samples=num_samples,
        linplugin=linplugin,
        empirical_fisher=empirical_fisher,
        learning_rate=learning_rate,
    )

    bog_result, _ = run_rebayes_algorithm(key, fg_bog, inputs, labels)
    return bog_result.mean, bog_result.cov

def fg_bbb(
        init_param_mean: Array,
        unravel_fn: callable,
        apply_fn: callable,
        inputs: Array,
        labels: Array,
        key: Array,
        init_param_cov: float | Array,
        log_likelihood: callable,
        obs_function: callable,
        obs_cov: Array,
        dynamics_decay: float = 1.0,
        process_noise: float = 0.0,
        num_samples: int = 10,
        linplugin: bool = True,
        empirical_fisher: bool = False,
        learning_rate: float = 0.1,
        num_iter: int = 10,
        **kwargs
    ) -> tuple[Array, Array]:
    """Train a neural network using the full-covariance Gaussian Bayes-by-backprop (BBB) algorithm.

    Args:
        init_param_mean (Array): Initial parameters.
        unravel_fn (callable): Parameter unravel function.
        apply_fn (callable): Model apply function.
        inputs (Array): Model input(s).
        labels (Array): Corresponding label(s).
        key (Array): JAX PRNG Key.
        init_param_cov (float | Array): Initial parameter covariance.
        log_likelihood (callable): Log-likelihood function.
        obs_function (callable, optional): Observation function. Defaults to identity function.
        obs_cov (Array): Observation covariance.
        dynamics_decay (float, optional): Parameter dynamics decay. Defaults to 1.0.
        process_noise (float, optional): Parameter process noise. Defaults to 0.0.
        num_samples (int, optional): Number of samples to use for each update. Defaults to 10.
        linplugin (bool, optional): Whether to use the linearized plugin method. Defaults to True.
        empirical_fisher (bool, optional): Whether to use the empirical Fisher approximation to the Hessian matrix. Defaults to False.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.1.
        num_iter (int, optional): Number of iterations per step. Defaults to 10.
    """
    init_param_cov = init_param_cov if isinstance(init_param_cov, Array) else init_param_cov * jnp.eye(len(init_param_mean))

    fg_bog = bbb.fg_bbb(
        init_mean=init_param_mean,
        init_cov=init_param_cov,
        log_likelihood=log_likelihood,
        emission_mean_function=lambda w, x: obs_function(apply_fn(unravel_fn(w), x)),
        emission_cov_function=lambda w, x: obs_cov,
        dynamics_decay=dynamics_decay,
        process_noise=process_noise,
        num_samples=num_samples,
        linplugin=linplugin,
        empirical_fisher=empirical_fisher,
        learning_rate=learning_rate,
        num_iter=num_iter,
    )

    bog_result, _ = run_rebayes_algorithm(key, fg_bog, inputs, labels)
    return bog_result.mean, bog_result.cov
