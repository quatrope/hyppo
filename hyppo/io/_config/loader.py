"""Configuration loaders for YAML and JSON formats."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from hyppo.core import FeatureSpace
from hyppo.extractor import registry


def load_config_yaml(config_path: str | Path) -> FeatureSpace:
    """
    Load YAML configuration and return FeatureSpace.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured FeatureSpace ready for extraction

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or configuration is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        config_dict = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")

    json_str = json.dumps(config_dict)

    return load_config_json_str(json_str)


def load_config_json(config_path: str | Path) -> FeatureSpace:
    """
    Load JSON configuration and return FeatureSpace.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configured FeatureSpace ready for extraction

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If JSON is invalid or configuration is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    return load_config_json_str(content)


def load_config_json_str(json_str: str) -> FeatureSpace:
    """
    Load JSON string configuration and return FeatureSpace.

    Args:
        json_str: JSON string containing configuration

    Returns:
        Configured FeatureSpace ready for extraction

    Raises:
        ValueError: If JSON is invalid or configuration is malformed
    """
    try:
        config_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    return _build_feature_space(config_dict)


def _build_feature_space(config_dict: Dict[str, Any]) -> FeatureSpace:
    """
    Build FeatureSpace from configuration dictionary.

    Args:
        config_dict: Dictionary containing pipeline configuration

    Returns:
        Configured FeatureSpace

    Raises:
        ValueError: If configuration is malformed
    """
    if "pipeline" not in config_dict:
        raise ValueError(
            "Required field 'pipeline' missing from configuration"
        )

    pipeline = config_dict["pipeline"]

    if not isinstance(pipeline, dict):
        raise ValueError("Field 'pipeline' must be a dictionary")

    if not pipeline:
        raise ValueError("Pipeline cannot be empty")

    extractor_configs = {}

    for feature_name, extractor_spec in pipeline.items():
        if not isinstance(extractor_spec, dict):
            raise ValueError(
                f"Extractor '{feature_name}' specification must be a dictionary"
            )

        if "extractor" not in extractor_spec:
            raise ValueError(
                f"Required field 'extractor' missing for '{feature_name}'"
            )

        extractor_type = extractor_spec["extractor"]

        if not registry.is_registered(extractor_type):
            raise ValueError(f"Unknown extractor type: {extractor_type}")

        extractor_class = registry.get(extractor_type)

        params = extractor_spec.get("params", {})

        if not isinstance(params, dict):
            raise ValueError(
                f"Field 'params' for extractor '{feature_name}' must be a dictionary"
            )

        try:
            extractor_instance = extractor_class(**params)
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate {extractor_type} with parameters "
                f"{params}: {e}"
            )

        extractor_configs[feature_name] = (extractor_instance, {})

    return FeatureSpace(extractor_configs)
