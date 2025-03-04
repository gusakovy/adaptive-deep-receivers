import unittest
import jax
import jax.random as jr
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from adr import DeepSICBlock, DeepSIC


class TestDeepSICBlock(unittest.TestCase):
    def setUp(self):
        self.key = jr.PRNGKey(42)
        self.deepsic_block = DeepSICBlock(
            symbol_bits=2,
            num_users=3,
            num_antennas=4,
            hidden_dim=8
        )
        self.rx_size = (self.deepsic_block.num_antennas
                        if self.deepsic_block.symbol_bits == 1
                        else 2 * self.deepsic_block.num_antennas)
        self.input_size = (self.rx_size
                           + self.deepsic_block.symbol_bits
                           * (self.deepsic_block.num_users - 1))

    def test_param_init(self):
        params = self.deepsic_block.init(self.key, jnp.empty((1, self.input_size)))
        self.assertIsNotNone(params)
        flat_params, _ = ravel_pytree(params)
        expected_num_params = (self.input_size * self.deepsic_block.hidden_dim + self.deepsic_block.hidden_dim + 
                               self.deepsic_block.hidden_dim * self.deepsic_block.symbol_bits + self.deepsic_block.symbol_bits)
        self.assertEqual(flat_params.shape[0], expected_num_params)
    
    def test_inference(self):
        inputs = jnp.stack([
            jnp.ones(self.input_size),
            -jnp.ones(self.input_size)
        ])
        params = self.deepsic_block.init(self.key, jnp.empty((1, self.input_size)))
        outputs = self.deepsic_block.apply(params, inputs)
        self.assertEqual(outputs.shape, (inputs.shape[0], self.deepsic_block.symbol_bits))


class TestDeepSIC(unittest.TestCase):
    def setUp(self):
        self.deepsic_model = DeepSIC(
            key=jr.PRNGKey(42),
            symbol_bits=2,
            num_users=3,
            num_antennas=4,
            num_layers=5,
            hidden_dim=8,
        )
        self.rx = jnp.stack([
            jnp.ones(self.deepsic_model.rx_size),
            -jnp.ones(self.deepsic_model.rx_size)
        ])

    def test_inference(self):
        soft_decisions = self.deepsic_model.soft_decode(self.rx)
        self.assertEqual(soft_decisions.shape[0], self.rx.shape[0])
        self.assertEqual(soft_decisions.shape[1], self.deepsic_model.num_users)
        self.assertEqual(soft_decisions.shape[2], self.deepsic_model.symbol_bits)

    def test_batched_inference(self):
        batch_rx = jnp.stack([self.rx, 2 * self.rx, 3 * self.rx])
        batch_soft_decisions = jax.vmap(self.deepsic_model.soft_decode)(batch_rx)
        self.assertEqual(batch_soft_decisions.shape[0], batch_rx.shape[0])
        self.assertEqual(batch_soft_decisions.shape[1], self.rx.shape[0])
        self.assertEqual(batch_soft_decisions.shape[2], self.deepsic_model.num_users)
        self.assertEqual(batch_soft_decisions.shape[3], self.deepsic_model.symbol_bits)

    def test_inference_consistency(self):
        soft_decisions = self.deepsic_model.soft_decode(self.rx)
        soft_decisions_again = self.deepsic_model.soft_decode(self.rx)
        self.assertTrue(jnp.allclose(soft_decisions, soft_decisions_again))

    def test_partial_consistency(self):
        soft_decisions_two_layers = self.deepsic_model.soft_decode(self.rx, first_layers=2)
        soft_decisions_four_layers = self.deepsic_model.soft_decode(self.rx, first_layers=4)
        self.assertFalse(jnp.allclose(soft_decisions_two_layers, soft_decisions_four_layers))
