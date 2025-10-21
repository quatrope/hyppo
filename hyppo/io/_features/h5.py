import h5py
from hyppo.core import FeatureCollection
import numpy as np
from pathlib import Path


def save_feature_collection(
    collection: FeatureCollection,
    path: Path | str,
) -> None:
    """
    Save FeatureCollection to HDF5 file.

    Stores feature data arrays and metadata in a structured HDF5 format with
    separate groups for features and metadata.

    Args:
        collection: FeatureCollection to save
        path: Output file path (must have .h5 extension)

    Raises:
        ValueError: If path doesn't have .h5 extension
        ValueError: If collection is empty

    Example:
        >>> results = fs.extract(hsi)
        >>> save_feature_collection(results, "output.h5")
    """
    if not isinstance(path, Path):
        path = Path(path)

    if path.suffix != ".h5":
        raise ValueError(f"Path must have .h5 extension, got {path.suffix}")

    if len(collection) == 0:
        raise ValueError("Cannot save empty FeatureCollection")

    with h5py.File(path, "w") as f:
        features_group = f.create_group("features")
        metadata_group = f.create_group("metadata")

        for feature_name, feature in collection.items():
            feature_subgroup = features_group.create_group(feature_name)
            metadata_subgroup = metadata_group.create_group(feature_name)

            if hasattr(feature, "data") and isinstance(feature.data, dict):
                for key, value in feature.data.items():
                    if isinstance(value, np.ndarray):
                        feature_subgroup.create_dataset(key, data=value)

            if hasattr(feature, "extractor") and feature.extractor is not None:
                metadata_subgroup.attrs["extractor_type"] = type(
                    feature.extractor
                ).__name__

            if hasattr(feature, "inputs_used"):
                metadata_subgroup.attrs["inputs_used"] = str(feature.inputs_used)

            if hasattr(feature, "data") and isinstance(feature.data, dict):
                metadata_subgroup.attrs["feature_keys"] = list(feature.data.keys())
