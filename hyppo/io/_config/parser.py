import json
import yaml
from pathlib import Path
from typing import Dict, Any

from .models import Config, PipelineConfig, ExtractorConfig


def parse_json(data: str | Dict[str, Any]) -> Config:
    """
    Parse JSON configuration data into Config object.

    Args:
        data: JSON string or dictionary containing configuration

    Returns:
        Config object representing the parsed configuration

    Raises:
        ValueError: If configuration format is invalid
        KeyError: If required fields are missing
    """
    if isinstance(data, str):
        try:
            config_dict = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        config_dict = data

    return _parse_config_dict(config_dict)


def parse_yaml(data: str) -> Config:
    """
    Parse YAML configuration data by converting to JSON then parsing.

    Args:
        data: YAML string containing configuration

    Returns:
        Config object representing the parsed configuration

    Raises:
        ValueError: If YAML or configuration format is invalid
    """
    try:
        config_dict = yaml.safe_load(data)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")

    return parse_json(config_dict)


def parse_config(config_path: str | Path) -> Config:
    """
    Parse configuration file automatically detecting format (JSON or YAML).

    Args:
        config_path: Path to configuration file

    Returns:
        Config object representing the parsed configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is unsupported or invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    suffix = config_path.suffix.lower()

    if suffix == ".json":
        return parse_json(content)
    elif suffix in [".yaml", ".yml"]:
        return parse_yaml(content)
    else:
        try:
            return parse_json(content)
        except ValueError:
            try:
                return parse_yaml(content)
            except ValueError:
                raise ValueError(
                    f"Unsupported file format '{suffix}'. "
                    f"Supported formats: .json, .yaml, .yml"
                )


def _parse_config_dict(config_dict: Dict[str, Any]) -> Config:
    """
    Parse configuration dictionary into Config object.

    Args:
        config_dict: Dictionary containing configuration data

    Returns:
        Config object

    Raises:
        KeyError: If required fields are missing
        ValueError: If configuration format is invalid
    """
    if "pipeline" not in config_dict:
        raise KeyError("Required field 'pipeline' missing from configuration")

    # Parse pipeline
    pipeline_dict = config_dict["pipeline"]
    if not isinstance(pipeline_dict, dict):
        raise ValueError("Field 'pipeline' must be a dictionary")

    pipeline = PipelineConfig()

    for extractor_name, extractor_spec in pipeline_dict.items():
        if not isinstance(extractor_spec, dict):
            raise ValueError(
                f"Extractor '{extractor_name}' specification must be a dictionary"
            )

        # Parse extractor specification
        if "extractor" not in extractor_spec:
            raise KeyError(f"Required field 'extractor' missing for '{extractor_name}'")

        extractor = extractor_spec["extractor"]
        inputs = extractor_spec.get("input", {})

        # Validate inputs format
        if not isinstance(inputs, dict):
            raise ValueError(
                f"Field 'input' for extractor '{extractor_name}' must be a dictionary"
            )

        extractor_config = ExtractorConfig(extractor=extractor, inputs=inputs)
        pipeline.add_extractor(extractor_name, extractor_config)

    return Config(pipeline=pipeline)
