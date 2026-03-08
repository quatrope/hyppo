"""Configuration loaders for YAML and JSON formats."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from hyppo.core import FeatureSpace
from hyppo.extractor import registry
from hyppo.runner import registry as runner_registry

from .config import Config


def load_config_yaml(config_path: str | Path) -> Config:
    """
    Load YAML configuration and return Config.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object containing FeatureSpace and BaseRunner

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


def load_config_json(config_path: str | Path) -> Config:
    """
    Load JSON configuration and return Config.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Config object containing FeatureSpace and BaseRunner

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


def load_config_json_str(json_str: str) -> Config:
    """
    Load JSON string configuration and return Config.

    Args:
        json_str: JSON string containing configuration

    Returns:
        Config object containing FeatureSpace and BaseRunner

    Raises:
        ValueError: If JSON is invalid or configuration is malformed
    """
    try:
        config_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    return _build_config(config_dict)


def _validate_pipeline(config_dict: Dict[str, Any]) -> dict:
    """Validate and extract pipeline from config."""
    if "pipeline" not in config_dict:
        raise ValueError(
            "Required field 'pipeline' missing from configuration"
        )

    pipeline = config_dict["pipeline"]

    if not isinstance(pipeline, dict):
        raise ValueError("Field 'pipeline' must be a dictionary")

    if not pipeline:
        raise ValueError("Pipeline cannot be empty")

    return pipeline


def _validate_extractor_spec(feature_name: str, extractor_spec) -> tuple[str, dict]:
    """Validate extractor spec structure and return type and params."""
    if not isinstance(extractor_spec, dict):
        raise ValueError(
            f"Extractor '{feature_name}' specification must be a "
            f"dictionary"
        )

    if "extractor" not in extractor_spec:
        raise ValueError(
            f"Required field 'extractor' missing for '{feature_name}'"
        )

    extractor_type = extractor_spec["extractor"]

    if not registry.is_registered(extractor_type):
        raise ValueError(f"Unknown extractor type: {extractor_type}")

    params = extractor_spec.get("params", {})

    if not isinstance(params, dict):
        raise ValueError(
            f"Field 'params' for extractor '{feature_name}' must "
            f"be a dictionary"
        )

    return extractor_type, params


def _process_extractor_spec(feature_name: str, extractor_spec: dict):
    """Validate and instantiate a single extractor from its spec."""
    extractor_type, params = _validate_extractor_spec(
        feature_name, extractor_spec
    )
    extractor_class = registry.get(extractor_type)

    try:
        return extractor_class(**params)
    except TypeError as e:
        raise ValueError(
            f"Failed to instantiate {extractor_type} with parameters "
            f"{params}: {e}"
        )


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
    pipeline = _validate_pipeline(config_dict)

    extractor_configs = {}
    for feature_name, extractor_spec in pipeline.items():
        extractor_instance = _process_extractor_spec(
            feature_name, extractor_spec
        )
        extractor_configs[feature_name] = (extractor_instance, {})

    return FeatureSpace(extractor_configs)


def _validate_runner_config(runner_config: dict) -> tuple[str, dict]:
    """Validate runner config and return type and params."""
    if not isinstance(runner_config, dict):
        raise ValueError("Field 'runner' must be a dictionary")

    if "type" not in runner_config:
        raise ValueError(
            "Required field 'type' missing from runner configuration"
        )

    runner_type = runner_config["type"]

    if not isinstance(runner_type, str):
        raise ValueError("Runner 'type' must be a string")

    params = runner_config.get("params", {})

    if not isinstance(params, dict):
        raise ValueError("Field 'params' for runner must be a dictionary")

    return runner_type, params


def _build_runner(config_dict: Dict[str, Any]):
    """
    Build BaseRunner from configuration dictionary.

    Args:
        config_dict: Dictionary containing runner configuration

    Returns:
        Configured BaseRunner instance

    Raises:
        ValueError: If runner configuration is malformed
    """
    runner_config = config_dict.get("runner", {"type": "sequential"})
    runner_type, params = _validate_runner_config(runner_config)

    try:
        return runner_registry.get(runner_type, params)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to create runner: {e}")


def _build_config(config_dict: Dict[str, Any]) -> Config:
    """
    Build Config from configuration dictionary.

    Args:
        config_dict: Dictionary containing complete configuration

    Returns:
        Config object with FeatureSpace and BaseRunner

    Raises:
        ValueError: If configuration is malformed
    """
    feature_space = _build_feature_space(config_dict)
    runner = _build_runner(config_dict)

    return Config(feature_space=feature_space, runner=runner)
