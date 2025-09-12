from abc import ABC, abstractmethod
import re
from typing import Any, Dict, Optional, Type
from dataclasses import dataclass
from hyppo.core import HSI


@dataclass
class InputDependency:
    """Specification for a named input dependency with type validation."""

    name: str  # Local name of the dependency (e.g., "mean", "mnf")
    extractor_type: Type["Extractor"]  # Expected type of the extractor
    required: bool = True  # Whether this input is required or has a default
    default_config: Optional[Dict[str, Any]] = (
        None  # Configuration for default instance
    )


class Extractor(ABC):
    """
    Unified base class for all feature extractors.

    This class provides:
    - Named input dependency declaration with type validation
    - Helper methods for different extraction patterns
    - Automatic feature naming
    - Result validation
    - Automatic registration in ExtractorRegistry
    """

    # Class-level input dependencies declaration
    input_dependencies: Dict[str, InputDependency] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the ExtractorRegistry."""
        super().__init_subclass__(**kwargs)

        # Avoid circular import by importing the registry here
        from .registry import registry

        registry.register(cls)

    @abstractmethod
    def extract(self, data: HSI, **inputs) -> Dict[str, Any]:
        """
        Extract features from the hyperspectral image.

        Args:
            data: HSI object containing the hyperspectral image
            **inputs: Named input results from dependency extractors

        Returns:
            Dictionary containing extracted features and metadata
        """
        pass

    @classmethod
    def get_input_dependencies(cls) -> Dict[str, InputDependency]:
        """Get declared input dependencies for this extractor."""
        return cls.input_dependencies.copy()

    @classmethod
    def get_default_for_input(cls, input_name: str) -> Optional["Extractor"]:
        """Create a default extractor instance for an optional input."""
        if input_name in cls.input_dependencies:
            dep = cls.input_dependencies[input_name]
            if not dep.required and dep.default_config is not None:
                return dep.extractor_type(**dep.default_config)
        return None

    # === Validation and Metadata ===

    def validate(self) -> None:
        """Validate extractor configuration."""
        pass

    @classmethod
    def feature_name(cls) -> str:
        """Generate feature name from class name."""
        name = cls.__name__
        name = name.removesuffix("FeatureExtractor").removesuffix("Extractor")
        name = re.split("(?<=.)(?=[A-Z])", name)
        return "_".join(name).lower()
