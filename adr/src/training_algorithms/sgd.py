import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from flax.training.train_state import TrainState
import optax
from optax import GradientTransformation

def minibatch_sgd(
        init_params: Array,
        unravel_fn: callable,
        apply_fn: callable,
        inputs: Array,
        labels: Array,
        loss_fn: callable,
        num_epochs: int,
        batch_size: int = None,
        shuffle: bool = True,
        optimizer: GradientTransformation = optax.adam(0.001),
        **kwargs
    ) -> Array:
    """Train a neural network using minibatch stochastic gradient descent (SGD).

    Args:
        init_param (Array): Initial parameters.
        unravel_fn (callable): Parameter unravel function.
        apply_fn (callable): Model apply function.
        inputs (Array): Model input(s).
        labels (Array): Corresponding label(s).
        loss_fn (callable): Loss function.
        num_epochs (int): Number of training epochs.
        batch_size (int, optional): Batch size for training. Defaults to None.
        shuffle (bool, optional): Shuffle data before training (only if batch_size is not None). Defaults to True.
        optimizer (GradientTransformation, optional): Optimizer. Defaults to optax.adam(0.001).
    """

    params = unravel_fn(init_params)
    state = TrainState.create(apply_fn=apply_fn, params=params, tx=optimizer)

    @jax.jit
    def train_step(state, inputs, labels):
        def loss_fn_of_params(params):
            outputs = apply_fn(params, inputs)
            loss = loss_fn(outputs, labels)
            return jnp.mean(loss)

        grads = jax.grad(loss_fn_of_params)(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    for epoch in range(num_epochs):
        if batch_size:
            if shuffle:
                perm = jr.permutation(jr.PRNGKey(epoch), inputs.shape[0])
                inputs = inputs[perm]
                labels = labels[perm]

            num_samples = inputs.shape[0]
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_inputs = inputs[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                state = train_step(state, batch_inputs, batch_labels)
        else:
            state = train_step(state, inputs, labels)

    new_params, _ = ravel_pytree(state.params)
    return new_params

def streaming_gd(
        init_params: Array,
        unravel_fn: callable,
        apply_fn: callable,
        inputs: Array,
        labels: Array,
        loss_fn: callable,
        learning_rate: float = 0.001,
        **kwargs
    ) -> Array:
    """Train a neural network from using gradient descent on streaming data.

    Args:
        init_param (Array): Initial parameters.
        unravel_fn (callable): Parameter unravel function.
        apply_fn (callable): Model apply function.
        inputs (Array): Model input(s).
        labels (Array): Corresponding label(s).
        loss_fn (callable): Loss function.
        learning_rate (float, optional): Learning rate. Defaults to 0.001.
    """

    params = unravel_fn(init_params)
    state = TrainState.create(apply_fn=apply_fn, params=params, tx=optax.adam(learning_rate))

    @jax.jit
    def train_step(state, input, label):
        def loss_fn_of_params(params):
            output = apply_fn(params, input)
            loss = loss_fn(output, label)
            return jnp.mean(loss)

        grads = jax.grad(loss_fn_of_params)(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    def step_fn(state, data):
        input, label = data
        state = train_step(state, input, label)
        return state, None

    data_stream = jax.tree.map(lambda x: x[:, None], (inputs, labels))
    state, _ = jax.lax.scan(step_fn, state, data_stream)

    new_params, _ = ravel_pytree(state.params)
    return new_params