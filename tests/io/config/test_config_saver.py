"""Tests for FeatureSpace configuration saving functionality."""

import json
from pathlib import Path

import pytest
import yaml

from hyppo import io
from hyppo.core import FeatureSpace, HSI
from hyppo.extractor import (
    NDVIExtractor,
    PCAExtractor,
    SAVIExtractor,
)
from hyppo.extractor.base import Extractor


class ExtractorWithUnderscoreParam(Extractor):
    """Test extractor with underscore parameter."""

    def __init__(self, _internal_param=None, normal_param=1):
        """Initialize test extractor."""
        self._internal_param = _internal_param
        self.normal_param = normal_param

    def _extract(self, data: HSI, **inputs) -> dict:
        return {"features": data.reflectance.mean(axis=2)}


class ExtractorWithNoneDefault(Extractor):
    """Test extractor with None default parameter."""

    def __init__(self, optional_param=None, required_param=5):
        """Initialize test extractor."""
        self.optional_param = optional_param
        self.required_param = required_param

    def _extract(self, data: HSI, **inputs) -> dict:
        return {"features": data.reflectance.mean(axis=2)}


class TestSaveConfigYAML:
    """Test cases for save_config_yaml function."""

    def test_save_config_yaml_creates_file(self, tmp_path):
        """Test that save_config_yaml creates a YAML file."""
        # Arrange: Create FeatureSpace
        fs = FeatureSpace.from_list([NDVIExtractor(), SAVIExtractor()])
        yaml_path = tmp_path / "config.yaml"

        # Act: Save configuration
        io.save_config_yaml(fs, yaml_path)

        # Assert: File exists
        assert yaml_path.exists()

    def test_save_config_yaml_invalid_extension(self, tmp_path):
        """Test save_config_yaml raises ValueError for invalid extension."""
        # Arrange: Create FeatureSpace and path with wrong extension
        fs = FeatureSpace.from_list([NDVIExtractor()])
        txt_path = tmp_path / "config.txt"

        # Act & Assert: Verify ValueError is raised
        with pytest.raises(ValueError, match=r"\.yaml.*\.yml"):
            io.save_config_yaml(fs, txt_path)

    def test_save_config_yaml_simple_pipeline(self, tmp_path):
        """Test saving simple pipeline to YAML."""
        # Arrange: Create FeatureSpace with simple extractors
        fs = FeatureSpace.from_list([NDVIExtractor(), SAVIExtractor()])
        yaml_path = tmp_path / "config.yaml"

        # Act: Save configuration
        io.save_config_yaml(fs, yaml_path)

        # Assert: Verify YAML structure
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        assert "pipeline" in config
        assert "ndvi" in config["pipeline"]
        assert "savi" in config["pipeline"]
        assert config["pipeline"]["ndvi"]["extractor"] == "NDVIExtractor"
        assert config["pipeline"]["savi"]["extractor"] == "SAVIExtractor"

    def test_save_config_yaml_with_parameters(self, tmp_path):
        """Test saving extractors with parameters."""
        # Arrange: Create FeatureSpace with extractor that has parameters
        fs = FeatureSpace.from_list([PCAExtractor(n_components=5)])
        yaml_path = tmp_path / "config.yaml"

        # Act: Save configuration
        io.save_config_yaml(fs, yaml_path)

        # Assert: Verify parameters are saved
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        feature_name = PCAExtractor.feature_name()
        assert feature_name in config["pipeline"]
        assert config["pipeline"][feature_name]["extractor"] == "PCAExtractor"
        assert "params" in config["pipeline"][feature_name]
        assert config["pipeline"][feature_name]["params"]["n_components"] == 5

    def test_save_config_yaml_accepts_yml_extension(self, tmp_path):
        """Test that save_config_yaml accepts .yml extension."""
        # Arrange: Create FeatureSpace
        fs = FeatureSpace.from_list([NDVIExtractor()])
        yml_path = tmp_path / "config.yml"

        # Act: Save configuration
        io.save_config_yaml(fs, yml_path)

        # Assert: File created
        assert yml_path.exists()

    def test_save_config_yaml_accepts_string_path(self, tmp_path):
        """Test that save_config_yaml accepts string paths."""
        # Arrange: Create FeatureSpace and string path
        fs = FeatureSpace.from_list([NDVIExtractor()])
        yaml_path = str(tmp_path / "config.yaml")

        # Act: Save configuration
        io.save_config_yaml(fs, yaml_path)

        # Assert: File created
        assert Path(yaml_path).exists()

    def test_save_config_yaml_accepts_path_object(self, tmp_path):
        """Test that save_config_yaml accepts Path objects."""
        # Arrange: Create FeatureSpace and Path object
        fs = FeatureSpace.from_list([NDVIExtractor()])
        yaml_path = Path(tmp_path / "config.yaml")

        # Act: Save configuration
        io.save_config_yaml(fs, yaml_path)

        # Assert: File created
        assert yaml_path.exists()

    def test_save_config_yaml_roundtrip(self, tmp_path):
        """Test saved YAML can be loaded back to equivalent FeatureSpace."""
        # Arrange: Create FeatureSpace with multiple extractors
        original_fs = FeatureSpace.from_list(
            [NDVIExtractor(), SAVIExtractor(), PCAExtractor()]
        )
        yaml_path = tmp_path / "config.yaml"

        # Act: Save and reload
        io.save_config_yaml(original_fs, yaml_path)
        loaded_config = io.load_config_yaml(yaml_path)

        # Assert: Same extractors
        assert set(loaded_config.feature_space.extractors.keys()) == set(
            original_fs.extractors.keys()
        )
        assert len(loaded_config.feature_space.extractors) == len(
            original_fs.extractors
        )

    def test_save_config_yaml_skip_underscore_params(self, tmp_path):
        """Test that parameters starting with underscore are skipped."""
        # Arrange: Create extractor with underscore parameter
        extractor = ExtractorWithUnderscoreParam(
            _internal_param="secret", normal_param=42
        )
        fs = FeatureSpace({"test": (extractor, {})})
        yaml_path = tmp_path / "config.yaml"

        # Act: Save configuration
        io.save_config_yaml(fs, yaml_path)

        # Assert: Underscore parameter not saved
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        params = config["pipeline"]["test"]["params"]
        assert "_internal_param" not in params
        assert "normal_param" in params
        assert params["normal_param"] == 42

    def test_save_config_yaml_skip_none_matching_default(self, tmp_path):
        """Test that None values matching None defaults are skipped."""
        # Arrange: Create extractor with None parameter matching default
        extractor = ExtractorWithNoneDefault(
            optional_param=None, required_param=10
        )
        fs = FeatureSpace({"test": (extractor, {})})
        yaml_path = tmp_path / "config.yaml"

        # Act: Save configuration
        io.save_config_yaml(fs, yaml_path)

        # Assert: None parameter not saved
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        params = config["pipeline"]["test"]["params"]
        assert "optional_param" not in params
        assert "required_param" in params
        assert params["required_param"] == 10


class TestSaveConfigJSON:
    """Test cases for save_config_json function."""

    def test_save_config_json_creates_file(self, tmp_path):
        """Test that save_config_json creates a JSON file."""
        # Arrange: Create FeatureSpace
        fs = FeatureSpace.from_list([NDVIExtractor(), SAVIExtractor()])
        json_path = tmp_path / "config.json"

        # Act: Save configuration
        io.save_config_json(fs, json_path)

        # Assert: File exists
        assert json_path.exists()

    def test_save_config_json_invalid_extension(self, tmp_path):
        """Test save_config_json raises ValueError for invalid extension."""
        # Arrange: Create FeatureSpace and path with wrong extension
        fs = FeatureSpace.from_list([NDVIExtractor()])
        txt_path = tmp_path / "config.txt"

        # Act & Assert: Verify ValueError is raised
        with pytest.raises(ValueError, match=r"\.json"):
            io.save_config_json(fs, txt_path)

    def test_save_config_json_simple_pipeline(self, tmp_path):
        """Test saving simple pipeline to JSON."""
        # Arrange: Create FeatureSpace with simple extractors
        fs = FeatureSpace.from_list([NDVIExtractor(), SAVIExtractor()])
        json_path = tmp_path / "config.json"

        # Act: Save configuration
        io.save_config_json(fs, json_path)

        # Assert: Verify JSON structure
        with open(json_path, "r") as f:
            config = json.load(f)

        assert "pipeline" in config
        assert "ndvi" in config["pipeline"]
        assert "savi" in config["pipeline"]
        assert config["pipeline"]["ndvi"]["extractor"] == "NDVIExtractor"
        assert config["pipeline"]["savi"]["extractor"] == "SAVIExtractor"

    def test_save_config_json_with_parameters(self, tmp_path):
        """Test saving extractors with parameters."""
        # Arrange: Create FeatureSpace with extractor that has parameters
        fs = FeatureSpace.from_list([PCAExtractor(n_components=10)])
        json_path = tmp_path / "config.json"

        # Act: Save configuration
        io.save_config_json(fs, json_path)

        # Assert: Verify parameters are saved
        with open(json_path, "r") as f:
            config = json.load(f)

        feature_name = PCAExtractor.feature_name()
        assert feature_name in config["pipeline"]
        assert config["pipeline"][feature_name]["extractor"] == "PCAExtractor"
        assert "params" in config["pipeline"][feature_name]
        assert config["pipeline"][feature_name]["params"]["n_components"] == 10

    def test_save_config_json_accepts_string_path(self, tmp_path):
        """Test that save_config_json accepts string paths."""
        # Arrange: Create FeatureSpace and string path
        fs = FeatureSpace.from_list([NDVIExtractor()])
        json_path = str(tmp_path / "config.json")

        # Act: Save configuration
        io.save_config_json(fs, json_path)

        # Assert: File created
        assert Path(json_path).exists()

    def test_save_config_json_accepts_path_object(self, tmp_path):
        """Test that save_config_json accepts Path objects."""
        # Arrange: Create FeatureSpace and Path object
        fs = FeatureSpace.from_list([NDVIExtractor()])
        json_path = Path(tmp_path / "config.json")

        # Act: Save configuration
        io.save_config_json(fs, json_path)

        # Assert: File created
        assert json_path.exists()

    def test_save_config_json_roundtrip(self, tmp_path):
        """Test saved JSON can be loaded back to equivalent FeatureSpace."""
        # Arrange: Create FeatureSpace with multiple extractors
        original_fs = FeatureSpace.from_list(
            [NDVIExtractor(), SAVIExtractor(), PCAExtractor(n_components=5)]
        )
        json_path = tmp_path / "config.json"

        # Act: Save and reload
        io.save_config_json(original_fs, json_path)
        loaded_config = io.load_config_json(json_path)

        # Assert: Same extractors
        assert set(loaded_config.feature_space.extractors.keys()) == set(
            original_fs.extractors.keys()
        )
        assert len(loaded_config.feature_space.extractors) == len(
            original_fs.extractors
        )


class TestConfigSaveMethod:
    """Test cases for Config.save() method."""

    def test_save_yaml_format(self, tmp_path):
        """Test save with .yaml extension."""
        # Arrange: Create Config
        from hyppo.io import Config

        fs = FeatureSpace.from_list([NDVIExtractor(), SAVIExtractor()])
        config = Config(feature_space=fs)
        yaml_path = tmp_path / "config.yaml"

        # Act: Call save method
        config.save(yaml_path)

        # Assert: File exists and is valid YAML
        assert yaml_path.exists()
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        assert "pipeline" in data

    def test_save_yml_format(self, tmp_path):
        """Test save with .yml extension."""
        # Arrange: Create Config
        from hyppo.io import Config

        fs = FeatureSpace.from_list([NDVIExtractor()])
        config = Config(feature_space=fs)
        yml_path = tmp_path / "config.yml"

        # Act: Call save method
        config.save(yml_path)

        # Assert: File exists
        assert yml_path.exists()

    def test_save_json_format(self, tmp_path):
        """Test save with .json extension."""
        # Arrange: Create Config
        from hyppo.io import Config

        fs = FeatureSpace.from_list([NDVIExtractor(), SAVIExtractor()])
        config = Config(feature_space=fs)
        json_path = tmp_path / "config.json"

        # Act: Call save method
        config.save(json_path)

        # Assert: File exists and is valid JSON
        assert json_path.exists()
        with open(json_path, "r") as f:
            data = json.load(f)
        assert "pipeline" in data

    def test_save_invalid_extension(self, tmp_path):
        """Test save raises ValueError for invalid extension."""
        # Arrange: Create Config and path with wrong extension
        from hyppo.io import Config

        fs = FeatureSpace.from_list([NDVIExtractor()])
        config = Config(feature_space=fs)
        txt_path = tmp_path / "config.txt"

        # Act & Assert: Verify ValueError is raised
        with pytest.raises(ValueError):
            config.save(txt_path)

    def test_save_accepts_string_path(self, tmp_path):
        """Test that save accepts string paths."""
        # Arrange: Create Config and string path
        from hyppo.io import Config

        fs = FeatureSpace.from_list([NDVIExtractor()])
        config = Config(feature_space=fs)
        yaml_path = str(tmp_path / "config.yaml")

        # Act: Save configuration
        config.save(yaml_path)

        # Assert: File created
        assert Path(yaml_path).exists()

    def test_save_roundtrip_yaml(self, tmp_path):
        """Test save and load_config_yaml roundtrip."""
        # Arrange: Create Config
        from hyppo.io import Config

        original_fs = FeatureSpace.from_list(
            [NDVIExtractor(), SAVIExtractor()]
        )
        original_config = Config(feature_space=original_fs)
        yaml_path = tmp_path / "config.yaml"

        # Act: Save and reload
        original_config.save(yaml_path)
        loaded_config = io.load_config_yaml(yaml_path)

        # Assert: Same extractors
        assert set(loaded_config.feature_space.extractors.keys()) == set(
            original_config.feature_space.extractors.keys()
        )

    def test_save_roundtrip_json(self, tmp_path):
        """Test save and load_config_json roundtrip."""
        # Arrange: Create Config
        from hyppo.io import Config

        original_fs = FeatureSpace.from_list(
            [NDVIExtractor(), PCAExtractor(n_components=3)]
        )
        original_config = Config(feature_space=original_fs)
        json_path = tmp_path / "config.json"

        # Act: Save and reload
        original_config.save(json_path)
        loaded_config = io.load_config_json(json_path)

        # Assert: Same extractors
        assert set(loaded_config.feature_space.extractors.keys()) == set(
            original_config.feature_space.extractors.keys()
        )


class TestFeatureSpaceSaveConfigMethod:
    """Test cases for FeatureSpace.save_config() method."""

    def test_save_config_yaml_format(self, tmp_path):
        """Test save_config with .yaml extension."""
        # Arrange: Create FeatureSpace
        fs = FeatureSpace.from_list([NDVIExtractor(), SAVIExtractor()])
        yaml_path = tmp_path / "config.yaml"

        # Act: Call save_config method
        fs.save_config(yaml_path)

        # Assert: File exists and is valid YAML
        assert yaml_path.exists()
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        assert "pipeline" in config

    def test_save_config_yml_format(self, tmp_path):
        """Test save_config with .yml extension."""
        # Arrange: Create FeatureSpace
        fs = FeatureSpace.from_list([NDVIExtractor()])
        yml_path = tmp_path / "config.yml"

        # Act: Call save_config method
        fs.save_config(yml_path)

        # Assert: File exists
        assert yml_path.exists()

    def test_save_config_json_format(self, tmp_path):
        """Test save_config with .json extension."""
        # Arrange: Create FeatureSpace
        fs = FeatureSpace.from_list([NDVIExtractor(), SAVIExtractor()])
        json_path = tmp_path / "config.json"

        # Act: Call save_config method
        fs.save_config(json_path)

        # Assert: File exists and is valid JSON
        assert json_path.exists()
        with open(json_path, "r") as f:
            config = json.load(f)
        assert "pipeline" in config

    def test_save_config_invalid_extension(self, tmp_path):
        """Test save_config raises ValueError for invalid extension."""
        # Arrange: Create FeatureSpace and path with wrong extension
        fs = FeatureSpace.from_list([NDVIExtractor()])
        txt_path = tmp_path / "config.txt"

        # Act & Assert: Verify ValueError is raised
        with pytest.raises(ValueError):
            fs.save_config(txt_path)

    def test_save_config_roundtrip_yaml(self, tmp_path):
        """Test save_config and load_config_yaml roundtrip."""
        # Arrange: Create FeatureSpace
        original_fs = FeatureSpace.from_list(
            [NDVIExtractor(), SAVIExtractor()]
        )
        yaml_path = tmp_path / "config.yaml"

        # Act: Save and reload
        original_fs.save_config(yaml_path)
        loaded_config = io.load_config_yaml(yaml_path)

        # Assert: Same extractors
        assert set(loaded_config.feature_space.extractors.keys()) == set(
            original_fs.extractors.keys()
        )

    def test_save_config_roundtrip_json(self, tmp_path):
        """Test save_config and load_config_json roundtrip."""
        # Arrange: Create FeatureSpace
        original_fs = FeatureSpace.from_list(
            [NDVIExtractor(), PCAExtractor(n_components=3)]
        )
        json_path = tmp_path / "config.json"

        # Act: Save and reload
        original_fs.save_config(json_path)
        loaded_config = io.load_config_json(json_path)

        # Assert: Same extractors
        assert set(loaded_config.feature_space.extractors.keys()) == set(
            original_fs.extractors.keys()
        )

    def test_save_config_with_string_path(self, tmp_path):
        """Test save_config accepts string path instead of Path object."""
        # Arrange: Create FeatureSpace and string path
        fs = FeatureSpace.from_list([NDVIExtractor()])
        yaml_path = str(tmp_path / "config.yaml")

        # Act: Call save_config with string
        fs.save_config(yaml_path)

        # Assert: File exists
        assert Path(yaml_path).exists()
