from abc import ABC, abstractmethod
from jax import Array


class Channel(ABC):
    """Base class for a channel."""
    
    @abstractmethod
    def transmit(self, s: Array, **kwargs) -> Array:
        """Simulate transmission of symbols.

        Args:
            s (Array): Symbols to be transmitted.
        """
        pass
