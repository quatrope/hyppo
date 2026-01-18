"""Configuration savers for YAML and JSON formats."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from hyppo.runner import registry as runner_registry

from .config import Config

if TYPE_CHECKING:
    from hyppo.core import FeatureSpace


def save_config_yaml(config: Config | "FeatureSpace", path: Path | str) -> None:
    """
    Save Config or FeatureSpace to YAML file.

    Args:
        config: Config or FeatureSpace instance to save
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

    config_dict = _build_config_dict(config)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def save_config_json(config: Config | "FeatureSpace", path: Path | str) -> None:
    """
    Save Config or FeatureSpace to JSON file.

    Args:
        config: Config or FeatureSpace instance to save
        path: Output file path (must have .json extension)

    Raises:
        ValueError: If path doesn't have .json extension
    """
    if not isinstance(path, Path):
        path = Path(path)

    if path.suffix != ".json":
        raise ValueError(f"Path must have .json extension, got {path.suffix}")

    config_dict = _build_config_dict(config)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def _build_config_dict(config: Config | "FeatureSpace") -> dict:
    """
    Build configuration dictionary from Config or FeatureSpace.

    Args:
        config: Config or FeatureSpace instance

    Returns:
        Configuration dictionary in format expected by loader
    """
    # Handle both Config and FeatureSpace
    if isinstance(config, Config):
        feature_space = config.feature_space
        runner = config.runner
    else:
        # Assume it's a FeatureSpace
        feature_space = config
        runner = None

    # Build pipeline configuration
    pipeline = {}
    for name, extractor in feature_space.extractors.items():
        extractor_type = type(extractor).__name__
        params = _extract_params(extractor)

        pipeline[name] = {
            "extractor": extractor_type,
            "params": params,
        }

    result = {"pipeline": pipeline}

    # Add runner configuration if present
    if runner is not None:
        runner_class = type(runner)
        runner_type = runner_registry.get_name(runner_class)
        if runner_type:
            result["runner"] = {"type": runner_type}

    return result


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

            # Skip attributes that look like internal state
            # (not constructor params)
            if param_name.startswith("_"):
                continue

            # Skip None values that match None defaults
            if value is None and param.default is None:
                continue

            params[param_name] = value

    return params
