from jax import Array
import jax.numpy as jnp
from bong.src import blr
from bong.util import run_rebayes_algorithm

def fg_blr(
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
    """Train a neural network using the full-covariance Gaussian Bayesian learning rule (BLR) algorithm.

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

    fg_blr = blr.fg_blr(
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
        num_iter=num_iter
    )

    blr_result, _ = run_rebayes_algorithm(key, fg_blr, inputs, labels)
    return blr_result.mean, blr_result.cov

def dg_blr(
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
    """Train a neural network using the diagonal-covariance Gaussian Bayesian learning rule (BLR) algorithm.

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

    init_param_cov = init_param_cov if isinstance(init_param_cov, Array) else init_param_cov * jnp.ones(len(init_param_mean))

    dg_blr = blr.dg_blr(
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
        num_iter=num_iter
    )

    blr_result, _ = run_rebayes_algorithm(key, dg_blr, inputs, labels)
    return blr_result.mean, blr_result.cov
