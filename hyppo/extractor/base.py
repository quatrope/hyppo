"""Base class for feature extractors."""

from abc import ABC, abstractmethod
from hyppo.core import HSI
import re


class Extractor(ABC):
    """Unified base class for all feature extractors."""

    def extract(self, data: HSI, **inputs):
        """Extract features from the hyperspectral image."""
        self._validate(data, **inputs)

        return self._extract(data, **inputs)

    def _validate(self, data: HSI, **inputs):
        """Validate extractor inputs."""
        pass

    @abstractmethod
    def _extract(self, data: HSI, **inputs) -> dict:
        """
        Extract features from the hyperspectral image.

        Args:
            data: HSI object containing the hyperspectral image
            **inputs: Named input results from dependency extractors

        Returns:
            Dictionary containing extracted features and metadata
        """

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Get declared input dependencies for this extractor."""
        return {}

    @classmethod
    def get_input_default(cls, input_name: str) -> "Extractor | None":
        """Create a default extractor instance for an optional input."""
        return None

    @classmethod
    def feature_name(cls):
        """Generate feature name from class name."""
        name = cls.__name__
        name = name.removesuffix("FeatureExtractor").removesuffix("Extractor")
        name = re.split("(?<=.)(?=[A-Z])", name)
        return "_".join(name).lower()
