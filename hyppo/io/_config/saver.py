"""Configuration savers for YAML and JSON formats."""

import json
import yaml
import inspect
from pathlib import Path

from hyppo.core import FeatureSpace


def save_config_yaml(feature_space: FeatureSpace, path: Path | str) -> None:
    """
    Save FeatureSpace configuration to YAML file.

    Args:
        feature_space: FeatureSpace instance to save
        path: Output file path (must have .yaml or .yml extension)

    Raises:
        ValueError: If path doesn't have .yaml or .yml extension
    """
    if not isinstance(path, Path):
        path = Path(path)

    if path.suffix not in [".yaml", ".yml"]:
        raise ValueError(
            f"Path must have .yaml or .yml extension, got {path.suffix}"
        )

    config_dict = _build_config_dict(feature_space)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def save_config_json(feature_space: FeatureSpace, path: Path | str) -> None:
    """
    Save FeatureSpace configuration to JSON file.

    Args:
        feature_space: FeatureSpace instance to save
        path: Output file path (must have .json extension)

    Raises:
        ValueError: If path doesn't have .json extension
    """
    if not isinstance(path, Path):
        path = Path(path)

    if path.suffix != ".json":
        raise ValueError(f"Path must have .json extension, got {path.suffix}")

    config_dict = _build_config_dict(feature_space)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def _build_config_dict(feature_space: FeatureSpace) -> dict:
    """
    Build configuration dictionary from FeatureSpace.

    Args:
        feature_space: FeatureSpace instance

    Returns:
        Configuration dictionary in format expected by loader
    """
    pipeline = {}

    for name, extractor in feature_space.extractors.items():
        extractor_type = type(extractor).__name__
        params = _extract_params(extractor)

        pipeline[name] = {
            "extractor": extractor_type,
            "params": params,
        }

    return {"pipeline": pipeline}


def _extract_params(extractor) -> dict:
    """
    Extract initialization parameters from an extractor instance.

    Args:
        extractor: Extractor instance

    Returns:
        Dictionary of parameter names and values
    """
    params = {}
    sig = inspect.signature(type(extractor).__init__)

    for param_name, param in sig.parameters.items():
        if param_name in ["self", "args", "kwargs"]:
            continue

        if hasattr(extractor, param_name):
            value = getattr(extractor, param_name)

            # Skip attributes that look like internal state (not constructor params)
            if param_name.startswith("_"):
                continue

            # Skip None values that match None defaults
            if value is None and param.default is None:
                continue

            params[param_name] = value

    return params
