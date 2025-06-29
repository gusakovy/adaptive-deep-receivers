from typing import Union
import jax
from jax import Array
from bong.src.states import AgentState
from bong.src import bong, blr, bog, bbb
from adr.src.utils import CovarianceType, TrainingMethod

def step_fn_builder(
    method: TrainingMethod,
    apply_fn: callable,
    obs_cov: Array,
    covariance_type: CovarianceType = CovarianceType.FULL,
    linplugin: bool = True,
    reparameterized: bool = False,
    dynamics_decay: float = 0.999,
    process_noise: Union[float, Array] = 0.0,
    log_likelihood: callable = None,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 0.1,
    num_iter: int = 10,
) -> callable:
    """Get a step function for the specified training method and configuration.

    Args:
        method (TrainingMethod): Core training method.
        apply_fn (callable): Model apply function.
        obs_cov (Array): Observation covariance matrix.
        covariance_type (CovarianceType): Type of covariance approximation for the parameters. Default is full.
        linplugin (bool, optional): Whether to use linearized plugin method. Default is True.
        reparameterized (bool, optional): Whether to use reparameterized version. Default is False.
        dynamics_decay (float, optional): Parameter dynamics decay factor (gamma). Default is 0.999.
        process_noise (Union[float, Array], optional): Process noise for parameter dynamics (Q). Default is 0.0.
        log_likelihood (callable, optional): Log likelihood function. Default is None.
        num_samples (int, optional): Number of samples for updates. Default is 10.
        empirical_fisher (bool, optional): Whether to use empirical Fisher approximation. Default is False.
        learning_rate (float, optional): Learning rate for BOG, BLR, and BBB methods. Default is 0.1.
        num_iter (int, optional): Number of iterations per step. Default is 10.
 
    Returns:
        callable:Step function with signature: (rng_key, params_mean, params_cov, input, target) -> (new_mean, new_cov, prediction).
    """
    predict_fn, update_fn = _get_predict_update_functions(
        method, covariance_type, linplugin, reparameterized
    )
    if log_likelihood is None:
        emission_mean_function = lambda w, x: jax.nn.sigmoid(apply_fn(w, x))
    else:
        emission_mean_function = lambda w, x: apply_fn(w, x)

    def _update_fn(carry, key):
        state, predicted_state, input, target = carry
        new_state = update_fn(
            rng_key=key,
            state_pred=predicted_state,
            state=state,
            x=input,
            y=target,
            log_likelihood=log_likelihood,
            emission_mean_function=emission_mean_function,
            emission_cov_function=lambda w, x: obs_cov,
            num_samples=num_samples,
            empirical_fisher=empirical_fisher,
            learning_rate=learning_rate,
            )
        new_carry = (new_state, predicted_state, input, target)
        return new_carry, None


    @jax.jit
    def step_fn(rng_key: Array, params_mean: Array, params_cov: Array, input: Array, target: Array) -> tuple[Array, Array, Array]:
        """Single training step for the specified method and configuration."""
        state = AgentState(params_mean, params_cov)
        predicted_state = predict_fn(state, dynamics_decay, process_noise)
        if method in [TrainingMethod.BONG, TrainingMethod.BOG]:
            state = predicted_state
        keys = jax.random.split(rng_key, num_iter)
        (updated_state, _, _, _), _ = jax.lax.scan(_update_fn, (state, predicted_state, input, target), keys)
        prediction = apply_fn(updated_state.mean, input)
        prediction = jax.nn.sigmoid(prediction)

        return updated_state.mean, updated_state.cov, prediction

    return step_fn

def _get_predict_update_functions(
    method: TrainingMethod, 
    covariance_type: CovarianceType, 
    linplugin: bool, 
    reparameterized: bool
) -> tuple[callable, callable]:
    """Get the appropriate predict and update functions based on method and configuration.

    Args:
        method (TrainingMethod): Core training method.
        covariance_type (CovarianceType): Type of covariance approximation for the parameters.
        linplugin (bool): Whether to use linearized plugin.
        reparameterized (bool): Whether to use reparameterized version.

    Returns:
        tuple: Tuple of (predict_function, update_function)
    """

    # Get the appropriate module
    if method == TrainingMethod.BONG:
        module = bong
    elif method == TrainingMethod.BLR:
        module = blr
    elif method == TrainingMethod.BOG:
        module = bog
    elif method == TrainingMethod.BBB:
        module = bbb
    else:
        raise ValueError(f"Unknown training method: {method.value}")

    # Get predict function
    predict_fn = getattr(module, f"predict_{method.value}")

    # Build update function
    update_fn_name = "update_"
    if linplugin:
        if covariance_type == CovarianceType.FULL:
            update_fn_name += "lfg_"
        elif covariance_type == CovarianceType.DG:
            update_fn_name += "ldg_"
    else:
        if covariance_type == CovarianceType.FULL:
            update_fn_name += "fg_"
        elif covariance_type == CovarianceType.DG:
            update_fn_name += "dg_"
    if reparameterized:
        update_fn_name += "reparam_"
    update_fn_name += method.value

    try:
        update_fn = getattr(module, update_fn_name)
    except AttributeError:
        raise ValueError(f"Update function '{update_fn_name}' not found in module {module.__name__}. "
                        f"This combination of parameters may not be supported.")

    return predict_fn, update_fn
