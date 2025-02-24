from abc import ABC, abstractmethod

class DetectorBase(ABC):
    """
    Base class for a detector.
    """

    @abstractmethod
    def soft_decode(self, rx):
        """Soft-decode a (batch of) symbol(s).

        Args:
            rx: Received signal(s).

        Returns:
            Probability vector(s) over the symbols in the constellation.
        """
        pass

    @abstractmethod
    def save(self, path):
        """Save the detector parameters to disk.

        Args:
            path: Save path for the detector parameters.
        """
        pass

    @abstractmethod
    def load(self, path):
        """Load the detector parameters from disk.

        Args:
            path: Path to detector parameters file.
        """
        pass
