"""Base runner interface for feature extraction execution engines."""

from abc import ABC, abstractmethod

from hyppo.core import FeatureCollection, FeatureSpace, HSI


class BaseRunner(ABC):
    """Abstract base class for feature extraction runners."""

    @abstractmethod
    def resolve(
        self, data: HSI, feature_space: FeatureSpace
    ) -> FeatureCollection:
        """Execute feature extraction workflow."""
        ...

    def _get_defaults_for_extractor(self, extractor) -> dict:
        """
        Get default extractors for optional inputs.

        Parameters
        ----------
        extractor
            The extractor to get defaults for

        Returns
        -------
        dict
            Dictionary mapping input names to default extractors
        """
        defaults = {}
        input_deps = extractor.get_input_dependencies()

        for input_name, dep_spec in input_deps.items():
            if not dep_spec["required"]:
                default_extractor = extractor.get_input_default(input_name)
                if default_extractor is not None:
                    defaults[input_name] = default_extractor

        return defaults
