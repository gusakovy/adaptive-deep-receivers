from flax import linen as nn


class DeepSICBlock(nn.Module):
    """Single block of a DeepSIC model.
    
    Args:
        symbol_bits (int): Number of bits per symbol.
        num_users (int): Number of users.
        num_antennas (int): Number of receive antennas.
        hidden_dim (int): Size of the hidden layer of the block.
        activation (callable, optional): Activation function. Defaults to ReLU.
    """
    symbol_bits: int
    num_users: int
    num_antennas: int
    hidden_dim: int
    activation: callable = nn.relu

    def setup(self):
        self.features = [self.hidden_dim, self.symbol_bits]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        return nn.Dense(self.features[-1])(x)