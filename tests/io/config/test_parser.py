import pytest
import json
import yaml
from pathlib import Path
from tempfile import NamedTemporaryFile

from hyppo.io import parse_json, parse_yaml, parse_config, Config


class TestParseJson:
    """Test JSON parsing functionality."""

    def test_parse_json_dict(self):
        """Test parsing JSON from dictionary."""
        json_dict = {"pipeline": {"std": {"extractor": "StdExtractor"}}}

        config = parse_json(json_dict)

        assert isinstance(config, Config)
        assert "std" in config.pipeline.extractors

    def test_parse_json_missing_required_fields(self):
        """Test parsing JSON with missing required fields."""

        with pytest.raises(KeyError, match="Required field 'pipeline' missing"):
            parse_json('{"TEST": "test.h5"}')


class TestParseYaml:
    """Test YAML parsing functionality."""

    def test_parse_yaml_simple(self):
        """Test parsing simple YAML configuration."""
        yaml_str = """
        pipeline:
          mean:
            extractor: MeanExtractor
        """

        config = parse_yaml(yaml_str)

        assert isinstance(config, Config)
        assert "mean" in config.pipeline.extractors

    def test_parse_yaml_complex(self):
        """Test parsing complex YAML with parameters and dependencies."""
        yaml_str = """
        pipeline:
          mean:
            extractor: MeanExtractor
          custom_feature:
            extractor:
              type: CustomExtractor
              param1: value1
              param2: 42
            input:
              mean_data: mean
        """

        config = parse_yaml(yaml_str)

        assert isinstance(config, Config)
        assert "mean" in config.pipeline.extractors
        assert "custom_feature" in config.pipeline.extractors

        custom_extractor = config.pipeline.extractors["custom_feature"]
        assert custom_extractor.extractor_type == "CustomExtractor"
        assert custom_extractor.extractor_params == {"param1": "value1", "param2": 42}
        assert custom_extractor.inputs == {"mean_data": "mean"}

    def test_parse_yaml_invalid_format(self):
        """Test parsing invalid YAML."""
        with pytest.raises(ValueError, match="Invalid YAML format"):
            parse_yaml("invalid: yaml: [unclosed")


class TestParseConfig:
    """Test configuration file parsing."""

    def test_parse_config_json_file(self):
        """Test parsing JSON configuration file."""
        config_data = {
            "input": "test.h5",
            "pipeline": {"mean": {"extractor": "MeanExtractor"}},
        }

        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            json_path = f.name

        try:
            config = parse_config(json_path)
            assert isinstance(config, Config)
            assert "mean" in config.pipeline.extractors
        finally:
            Path(json_path).unlink()

    def test_parse_config_yaml_file(self):
        """Test parsing YAML configuration file."""
        config_data = {
            "input": "test.h5",
            "pipeline": {"std": {"extractor": "StdExtractor"}},
        }

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name

        try:
            config = parse_config(yaml_path)
            assert isinstance(config, Config)
            assert "std" in config.pipeline.extractors
        finally:
            Path(yaml_path).unlink()

    def test_parse_config_file_not_found(self):
        """Test parsing non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            parse_config("non_existent_config.yaml")

    def test_parse_config_unsupported_format(self):
        """Test parsing file with unsupported format."""
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not a config")
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                parse_config(txt_path)
        finally:
            Path(txt_path).unlink()
