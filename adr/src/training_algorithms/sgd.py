from typing import NamedTuple
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from flax.training.train_state import TrainState
import optax
from optax import GradientTransformation

def build_sgd_train_fn(
        loss_fn: callable,
        dynamics_decay: float = 0.999,
        num_epochs: int = 10,
        batch_size: int = None,
        shuffle: bool = True,
        optimizer: GradientTransformation = optax.adam,
        learning_rate: float = 0.001,
        **kwargs
    ) -> callable:
    """Build a training function for stochastic gradient descent (SGD)."""

    def init_state(apply_fn, params):
        return TrainState.create(apply_fn=apply_fn, params=params, tx=optimizer(learning_rate))

    @jax.jit
    def train_fn(key, state, inputs, labels):
        keys = jr.split(key, num_epochs)

        def train_step(state, args):
            inputs, labels = args

            def loss_fn_of_params(params):
                outputs = state.apply_fn(params, inputs)
                loss = loss_fn(outputs, labels)
                return jnp.mean(loss)

            grads = jax.grad(loss_fn_of_params)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, None

        def epoch_step(carry, key):
            state, inputs, labels = carry

            if batch_size:
                if shuffle:
                    perm = jr.permutation(key, inputs.shape[0])
                    inputs = inputs[perm]
                    labels = labels[perm]

                state = state.replace(params=dynamics_decay*state.params)
                batched_inputs = jnp.reshape(inputs, (inputs.shape[0]//batch_size, batch_size, -1))
                batched_labels = jnp.reshape(labels, (labels.shape[0]//batch_size, batch_size, -1))
                state, _ = jax.lax.scan(train_step, state, (batched_inputs, batched_labels))
            else:
                state = state.replace(params=dynamics_decay*state.params)
                state, _ = train_step(state, (inputs, labels))

            return (state, inputs, labels), None

        (state, inputs, labels), _ = jax.lax.scan(epoch_step, (state, inputs, labels), keys)

        outputs = state.apply_fn(state.params, inputs)
        outputs = jax.nn.sigmoid(outputs)

        return state, outputs

    return init_state, train_fn

class GDState(NamedTuple):
    params: Array

def build_gd_step_fn(
        apply_fn: callable,
        loss_fn: callable,
        dynamics_decay: float = 0.999,
        num_iter: int = 10,
        learning_rate: float = 0.001,
        **kwargs
    ) -> callable:
    """Build a step function for gradient descent (GD)."""

    def init_state(apply_fn, params):
        return GDState(params)

    def predict_fn(state: GDState, dynamics_decay: float) -> GDState:
        return GDState(dynamics_decay * state.params)

    def update_fn(carry, _):
        state, input, target = carry

        def loss_fn_of_params(params):
            output = apply_fn(params, input)
            loss = loss_fn(output, target)
            return jnp.mean(loss)

        # Simple gradient descent update
        grad = jax.grad(loss_fn_of_params)(state.params)
        new_params = state.params - learning_rate * grad
        new_state = GDState(new_params)
        return (new_state, input, target), None

    @jax.jit
    def step_fn(state, input: Array, target: Array) -> tuple[GDState, Array]:
        """Single training step for the specified method and configuration."""
        predicted_state = predict_fn(state, dynamics_decay)
        (updated_state, _, _), _ = jax.lax.scan(update_fn, (predicted_state, input, target), jnp.arange(num_iter))
        prediction = apply_fn(updated_state.params, input)
        prediction = jax.nn.sigmoid(prediction)

        return updated_state, prediction

    return init_state, step_fn

def build_stateful_gd_step_fn(
        loss_fn: callable,
        dynamics_decay: float = 0.999,
        num_iter: int = 10,
        optimizer: GradientTransformation = optax.adam,
        learning_rate: float = 0.001,
        **kwargs
    ) -> callable:
    """Build a step function for stochastic gradient descent (SGD)."""

    def init_state(apply_fn, params):
        return TrainState.create(apply_fn=apply_fn, params=params, tx=optimizer(learning_rate))

    def update_fn(carry, _):
        state, input, target = carry

        def loss_fn_of_params(params):
            output = state.apply_fn(params, input)
            loss = loss_fn(output, target)
            return jnp.mean(loss)

        # Simple gradient descent update
        grads = jax.grad(loss_fn_of_params)(state.params)
        new_params = state.params - learning_rate * grads
        state = state.replace(params=new_params)
        return (state, input, target), None

    @jax.jit 
    def step_fn(state, input, target):
        # Apply dynamics decay
        state = state.replace(params=dynamics_decay*state.params)

        # Run multiple update iterations
        (state, _, _), _ = jax.lax.scan(update_fn, (state, input, target), jnp.arange(num_iter))

        # Get prediction
        prediction = state.apply_fn(state.params, input)
        prediction = jax.nn.sigmoid(prediction)

        return state, prediction

    return init_state, step_fn
