from typing import Any, Dict, Optional
import numpy as np

from hyppo.utils.attr_dict import AttrDict


class FeatureResult(AttrDict):
    """
    Dictionary for feature extraction results.

    Examples:
        >>> result = FeatureResult({'mean': [1, 2, 3], 'std': [0.1, 0.2, 0.3]})
        >>> result.mean
        [1, 2, 3]
        >>> result['mean']
        [1, 2, 3]
    """

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert all values to numpy arrays where possible."""
        result = {}
        for key, value in self.items():
            try:
                result[key] = np.array(value)
            except (ValueError, TypeError):
                result[key] = value
        return result


class FeatureResultCollection(AttrDict):
    """
    Collection of FeatureResult objects.

    This class manages results from multiple feature extractors

    Examples:
        >>> results = FeatureResultCollection()
        >>> results['mean'] = FeatureResult({'data': [1, 2, 3]})
        >>> results.mean.data
        [1, 2, 3]
        >>> results['mean']['data']
        [1, 2, 3]
    """

    def add_result(
        self,
        extractor_name: str,
        data: Dict[str, Any],
        extractor: Any = None,
        inputs_used: Optional[list] = None,
    ) -> None:
        """
        Add a feature extraction result.

        Args:
            extractor_name: Name of the extractor
            data: Extracted feature data
            extractor: The extractor instance used
            inputs_used: List of input names used by the extractor
        """
        result = FeatureResult(
            {"data": data, "extractor": extractor, "inputs_used": inputs_used or []}
        )
        self[extractor_name] = result

    def get_all_features(self) -> Dict[str, Any]:
        """Get all extracted feature data (without metadata)."""
        features = {}
        for extractor_name, result in self.items():
            if hasattr(result, "data") and isinstance(result.data, dict):
                features.update(result.data)
            else:
                features[extractor_name] = (
                    result.data if hasattr(result, "data") else result
                )
        return features

    def get_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about extractors and their usage."""
        metadata = {}
        for extractor_name, result in self.items():
            if isinstance(result, FeatureResult):
                metadata[extractor_name] = {
                    "extractor_type": (
                        type(result.extractor).__name__ if result.extractor else None
                    ),
                    "inputs_used": result.inputs_used,
                    "feature_keys": (
                        list(result.data.keys())
                        if isinstance(result.data, dict)
                        else []
                    ),
                }
        return metadata

    def get_extractor_names(self) -> list:
        """Get list of all extractor names."""
        return list(self.keys())

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to a regular nested dictionary."""
        return {
            name: (
                result.to_dict() if isinstance(result, FeatureResult) else dict(result)
            )
            for name, result in self.items()
        }


# results = FeatureResultCollection()





__all__ = ["FeatureResult", "FeatureResultCollection"]
