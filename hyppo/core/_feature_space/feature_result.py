import numpy as np
import pandas as pd

from hyppo.utils.bunch import Bunch


class FeatureResult(Bunch):
    """
    Dictionary for feature extraction results.

    Examples:
        >>> result = FeatureResult({'mean': [1, 2, 3], 'std': [0.1, 0.2, 0.3]})
        >>> result.mean
        [1, 2, 3]
        >>> result['mean']
        [1, 2, 3]
    """

    def __init__(self, data):
        super().__init__("FeatureResult", data)

    def to_numpy(self):
        """Convert all values to numpy arrays where possible."""
        result: dict[str, np.ndarray] = {}
        for key, value in self.items():
            try:
                result[key] = np.array(value)
            except (ValueError, TypeError):
                result[key] = value
        return result

    def describe(self):
        """
        Get summary information about this feature result.

        Returns:
            Dictionary with:
                - dimensions: Shape of the 'features' array if present
                - extra_data: Comma-separated list of extra data keys (excluding 'features')
        """
        data = self.get("data", {})

        features_shape = None
        if isinstance(data, dict) and "features" in data:
            features = data["features"]
            if isinstance(features, np.ndarray):
                features_shape = features.shape
            elif hasattr(features, "shape"):
                features_shape = features.shape

        extra_keys = []
        if isinstance(data, dict):
            extra_keys = [k for k in data.keys() if k != "features"]

        return {
            "dimensions": features_shape,
            "extra_data": ", ".join(extra_keys) if extra_keys else "",
        }


class FeatureResultCollection(Bunch):
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

    def __init__(self, data):
        super().__init__("FeatureResultCollection", data)

    def add_result(
        self,
        extractor_name: str,
        data: dict,
        extractor=None,
        inputs_used: list | None = None,
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
            {"data": data, "extractor": extractor, "inputs_used": inputs_used or []},
        )
        self[extractor_name] = result

    def get_all_features(self):
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

    def get_metadata(self):
        """Get metadata about extractors and their usage."""
        metadata: dict[str, dict] = {}
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

    def to_dict(self):
        """Convert to a regular nested dictionary."""
        return {
            name: (
                result.to_dict() if isinstance(result, FeatureResult) else dict(result)
            )
            for name, result in self.items()
        }

    def describe(self) -> pd.DataFrame:
        """
        Get summary information for all feature results.

        Returns:
            DataFrame with columns:
                - feature_name: Name of each feature
                - dimensions: Shape of the 'features' array
                - extra_data: Comma-separated list of extra data keys
        """
        rows = []
        for feature_name, result in self.items():
            if isinstance(result, FeatureResult):
                info = result.describe()
                info["feature_name"] = feature_name
                rows.append(info)

        return pd.DataFrame(rows, columns=["feature_name", "dimensions", "extra_data"])


__all__ = ["FeatureResult", "FeatureResultCollection"]
