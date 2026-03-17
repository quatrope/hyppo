import pandas as pd

from hyppo.utils.bunch import Bunch


class Feature(Bunch):
    """
    Dictionary for feature extraction results.

    Examples
    --------
    >>> result = Feature({'mean': [1, 2, 3], 'std': [0.1, 0.2, 0.3]})
    >>> result.mean
    [1, 2, 3]
    >>> result['mean']
    [1, 2, 3]
    """

    def __init__(self, data, extractor, inputs_used):
        """Initialize Feature with data, extractor, and inputs used."""
        mapping = {
            "result": data.get("features", None),
            "data": data,
            "extractor": extractor,
            "inputs_used": inputs_used,
        }
        super().__init__("Feature", mapping)

    @staticmethod
    def _get_features_shape(data):
        """Extract shape from features entry if available."""
        if not isinstance(data, dict) or "features" not in data:
            return None

        features = data["features"]
        if hasattr(features, "shape"):
            return features.shape
        return None

    def describe(self):
        """
        Get summary information about this feature result.

        Returns
        -------
        Dictionary with dimensions and extra_data keys
        """
        data = self.get("data", {})
        features_shape = self._get_features_shape(data)

        extra_keys = []
        if isinstance(data, dict):
            extra_keys = [k for k in data.keys() if k != "features"]

        return {
            "dimensions": features_shape,
            "extra_data": ", ".join(extra_keys) if extra_keys else "",
        }


class FeatureCollection(Bunch):
    """
    Collection of Feature objects.

    This class manages results from multiple feature extractors.

    Examples
    --------
    >>> results = FeatureCollection()
    >>> results['mean'] = Feature({'data': [1, 2, 3]})
    >>> results.mean.data
    [1, 2, 3]
    >>> results['mean']['data']
    [1, 2, 3]
    """

    def __init__(self, data: dict[str, Feature]):
        """Initialize FeatureCollection with dictionary of Feature objects."""
        super().__init__("FeatureCollection", data)

    @classmethod
    def from_features(
        cls, features: dict[str, Feature]
    ) -> "FeatureCollection":
        """
        Create a FeatureCollection from a dictionary of features.

        Parameters
        ----------
        features : Dictionary of features (extractor_name -> Feature)

        Returns
        -------
        FeatureCollection
        """
        return cls(features)

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
            if isinstance(result, Feature):
                metadata[extractor_name] = {
                    "extractor_type": (
                        type(result.extractor).__name__
                        if result.extractor
                        else None
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
                result.to_dict()
                if isinstance(result, Feature)
                else dict(result)
            )
            for name, result in self.items()
        }

    def describe(self) -> pd.DataFrame:
        """
        Get summary information for all feature results.

        Returns
        -------
        DataFrame with columns:
            - feature_name: Name of each feature
            - dimensions: Shape of the 'features' array
            - extra_data: Comma-separated list of extra data keys
        """
        rows = []
        for feature_name, result in self.items():
            info = result.describe()
            info["feature_name"] = feature_name
            rows.append(info)

        columns = ["feature_name", "dimensions", "extra_data"]
        return pd.DataFrame(rows, columns=columns)

    def save(self, path) -> None:
        """
        Save this FeatureCollection to HDF5 file.

        Parameters
        ----------
        path : Output file path (must have .h5 extension)

        Examples
        --------
        >>> results = fs.extract(hsi)
        >>> results.save("output.h5")
        """
        from hyppo import io

        io.save_feature_collection(self, path)


__all__ = ["Feature", "FeatureCollection"]
