import jax
import jax.numpy as jnp
import jax.random as jr
from flax.training.train_state import TrainState
import optax
from optax import GradientTransformation

def build_sgd_train_fn(
        dynamics_decay: float = 0.999,
        loss_fn: callable = None,
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

        for epoch in range(num_epochs):
            if batch_size:
                if shuffle:
                    perm = jr.permutation(keys[epoch], inputs.shape[0])
                    inputs = inputs[perm]
                    labels = labels[perm]

                state = state.replace(params=dynamics_decay*state.params)
                batched_inputs = jnp.reshape(inputs, (inputs.shape[0]//batch_size, batch_size, -1))
                batched_labels = jnp.reshape(labels, (labels.shape[0]//batch_size, batch_size, -1))
                state, _ = jax.lax.scan(train_step, state, (batched_inputs, batched_labels))
            else:
                state = state.replace(params=dynamics_decay*state.params)
                state, _ = train_step(state, (inputs, labels))

        outputs = state.apply_fn(state.params, inputs)
        return state, outputs

    return init_state, train_fn

def build_sgd_step_fn(
        dynamics_decay: float = 0.999,
        num_iter: int = 10,
        loss_fn: callable = None,
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

        grads = jax.grad(loss_fn_of_params)(state.params)
        state = state.apply_gradients(grads=grads)
        return (state, input, target), None

    @jax.jit
    def step_fn(state, input, target):

        state = state.replace(params=dynamics_decay*state.params)
        (state, _, _), _ = jax.lax.scan(update_fn, (state, input, target), jnp.arange(num_iter))
        
        prediction = state.apply_fn(state.params, input)
        prediction = jax.nn.sigmoid(prediction)

        return state, prediction

    return init_state, step_fn
