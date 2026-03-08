"""Tests for configuration loading and Config generation."""

import json
from pathlib import Path

import pytest
import yaml

from hyppo.core import FeatureSpace
from hyppo.io import Config, load_config_json, load_config_yaml
from hyppo.runner import BaseRunner, SequentialRunner


class TestConfigLoader:
    """Test configuration loading and FeatureSpace generation."""

    def test_load_yaml_simple_pipeline(self, tmp_path):
        """Test loading YAML config with simple pipeline."""
        # Arrange: Create YAML config file
        config_data = {
            "pipeline": {
                "n_d_v_i": {"extractor": "NDVIExtractor"},
                "s_a_v_i": {"extractor": "SAVIExtractor"},
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: FeatureSpace created correctly
        assert isinstance(config, Config)
        assert isinstance(config.feature_space, FeatureSpace)
        assert isinstance(config.runner, BaseRunner)
        assert len(config.feature_space.extractors) == 2
        assert "n_d_v_i" in config.feature_space.extractors
        assert "s_a_v_i" in config.feature_space.extractors

    def test_load_json_simple_pipeline(self, tmp_path):
        """Test loading JSON config with simple pipeline."""
        # Arrange: Create JSON config file
        config_data = {
            "pipeline": {
                "n_d_v_i": {"extractor": "NDVIExtractor"},
            }
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Act: Load configuration
        config = load_config_json(config_path)

        # Assert: FeatureSpace created correctly
        assert isinstance(config, Config)
        assert isinstance(config.feature_space, FeatureSpace)
        assert isinstance(config.runner, BaseRunner)
        assert "n_d_v_i" in config.feature_space.extractors

    def test_load_yaml_multiple_extractors(self, tmp_path):
        """Test loading YAML config with multiple extractors."""
        # Arrange: Create YAML with multiple extractors
        config_data = {
            "pipeline": {
                "n_d_v_i": {"extractor": "NDVIExtractor"},
                "s_a_v_i": {"extractor": "SAVIExtractor"},
                "p_c_a": {"extractor": "PCAExtractor"},
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: All extractors in pipeline
        assert isinstance(config, Config)
        assert isinstance(config.feature_space, FeatureSpace)
        assert isinstance(config.runner, BaseRunner)
        assert len(config.feature_space.extractors) == 3

    def test_load_extractor_with_parameters(self, tmp_path):
        """Test loading extractor with parameters."""
        # Arrange: Create config with extractor parameters
        config_data = {
            "pipeline": {
                "pca": {
                    "extractor": "PCAExtractor",
                    "params": {"n_components": 5},
                }
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: Extractor instantiated with parameters
        assert isinstance(config, Config)
        assert isinstance(config.feature_space, FeatureSpace)
        assert isinstance(config.runner, BaseRunner)
        assert "pca" in config.feature_space.extractors

    def test_load_multiple_extractors_with_parameters(self, tmp_path):
        """Test loading multiple extractors with various parameters."""
        # Arrange: Config with multiple extractors and parameters
        config_data = {
            "pipeline": {
                "n_d_v_i": {"extractor": "NDVIExtractor"},
                "s_a_v_i": {"extractor": "SAVIExtractor"},
                "pca": {
                    "extractor": "PCAExtractor",
                    "params": {"n_components": 10},
                },
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: All extractors created
        assert len(config.feature_space.extractors) == 3

    def test_missing_pipeline_raises_error(self, tmp_path):
        """Test that missing pipeline field raises error."""
        # Arrange: Config without pipeline field
        config_data = {"other_field": "value"}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="pipeline"):
            load_config_yaml(config_path)

    def test_empty_pipeline_raises_error(self, tmp_path):
        """Test that empty pipeline raises error."""
        # Arrange: Config with empty pipeline
        config_data = {"pipeline": {}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="empty"):
            load_config_yaml(config_path)

    def test_missing_extractor_field_raises_error(self, tmp_path):
        """Test that missing extractor field raises error."""
        # Arrange: Pipeline entry without extractor field
        config_data = {"pipeline": {"n_d_v_i": {"params": {}}}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="extractor"):
            load_config_yaml(config_path)

    def test_unknown_extractor_raises_error(self, tmp_path):
        """Test that unknown extractor type raises error."""
        # Arrange: Config with non-existent extractor
        config_data = {
            "pipeline": {"fake": {"extractor": "NonExistentExtractor"}}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="Unknown extractor"):
            load_config_yaml(config_path)

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML syntax raises error."""
        # Arrange: Create file with invalid YAML
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("pipeline:\n  mean: {extractor: NDVIExtractor\n")

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="YAML"):
            load_config_yaml(config_path)

    def test_invalid_json_raises_error(self, tmp_path):
        """Test that invalid JSON syntax raises error."""
        # Arrange: Create file with invalid JSON
        config_path = tmp_path / "invalid.json"
        with open(config_path, "w") as f:
            f.write('{"pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"')

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="JSON"):
            load_config_json(config_path)

    def test_nonexistent_yaml_file_raises_error(self):
        """Test that non-existent YAML file raises error."""
        # Arrange: Path to non-existent file
        config_path = Path("/nonexistent/config.yaml")

        # Act & Assert: Verify error raised
        with pytest.raises(FileNotFoundError):
            load_config_yaml(config_path)

    def test_nonexistent_json_file_raises_error(self):
        """Test that non-existent JSON file raises error."""
        # Arrange: Path to non-existent file
        config_path = Path("/nonexistent/config.json")

        # Act & Assert: Verify error raised
        with pytest.raises(FileNotFoundError):
            load_config_json(config_path)

    def test_automatic_dependency_resolution(self, tmp_path):
        """Test that dependencies are automatically resolved."""
        # Arrange: Config with multiple extractors
        config_data = {
            "pipeline": {
                "n_d_v_i": {"extractor": "NDVIExtractor"},
                "s_a_v_i": {"extractor": "SAVIExtractor"},
                "p_c_a": {"extractor": "PCAExtractor"},
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: FeatureSpace created with all extractors
        assert isinstance(config, Config)
        assert isinstance(config.feature_space, FeatureSpace)
        assert isinstance(config.runner, BaseRunner)
        assert len(config.feature_space.extractors) == 3

    def test_extract_from_loaded_config(self, tmp_path, sample_hsi):
        """Test feature extraction from loaded config."""
        # Arrange: Create config and load it
        config_data = {
            "pipeline": {
                "n_d_v_i": {"extractor": "NDVIExtractor"},
                "s_a_v_i": {"extractor": "SAVIExtractor"},
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        config = load_config_yaml(config_path)

        # Act: Execute extraction
        results = config.feature_space.extract(sample_hsi)

        # Assert: Results contain expected features
        assert "n_d_v_i" in results
        assert "s_a_v_i" in results

    @pytest.mark.parametrize(
        "loader,extension",
        [
            (load_config_yaml, ".yaml"),
            (load_config_json, ".json"),
        ],
    )
    def test_format_consistency(self, tmp_path, loader, extension):
        """Test that YAML and JSON produce identical FeatureSpace."""
        # Arrange: Same config in different formats
        config_data = {
            "pipeline": {
                "n_d_v_i": {"extractor": "NDVIExtractor"},
                "s_a_v_i": {"extractor": "SAVIExtractor"},
            }
        }
        config_path = tmp_path / f"config{extension}"

        if extension == ".yaml":
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
        else:
            with open(config_path, "w") as f:
                json.dump(config_data, f)

        # Act: Load configuration
        config = loader(config_path)

        # Assert: Same result regardless of format
        assert isinstance(config, Config)
        assert isinstance(config.feature_space, FeatureSpace)
        assert isinstance(config.runner, BaseRunner)
        assert len(config.feature_space.extractors) == 2

    def test_single_extractor_pipeline(self, tmp_path):
        """Test pipeline with single extractor."""
        # Arrange: Minimal pipeline
        config_data = {"pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: Single extractor loaded
        assert len(config.feature_space.extractors) == 1

    def test_extractor_name_with_special_characters(self, tmp_path):
        """Test extractor names with underscores and numbers."""
        # Arrange: Config with special names
        config_data = {
            "pipeline": {
                "mean_v1": {"extractor": "NDVIExtractor"},
                "std_2d": {"extractor": "SAVIExtractor"},
                "pca_3": {"extractor": "PCAExtractor"},
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: All names preserved
        assert "mean_v1" in config.feature_space.extractors
        assert "std_2d" in config.feature_space.extractors
        assert "pca_3" in config.feature_space.extractors

    def test_empty_params_dict(self, tmp_path):
        """Test extractor with empty params dict."""
        # Arrange: Config with empty params
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor", "params": {}}}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: Extractor created with defaults
        assert isinstance(config, Config)
        assert isinstance(config.feature_space, FeatureSpace)
        assert isinstance(config.runner, BaseRunner)

    def test_yaml_loads_via_json(self, tmp_path):
        """Test that YAML loading converts to dict then JSON."""
        # Arrange: Create YAML config
        config_data = {
            "pipeline": {
                "n_d_v_i": {"extractor": "NDVIExtractor"},
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load YAML configuration
        config = load_config_yaml(config_path)

        # Assert: Successfully loaded via YAML->dict->JSON path
        assert isinstance(config, Config)
        assert isinstance(config.feature_space, FeatureSpace)
        assert isinstance(config.runner, BaseRunner)
        assert "n_d_v_i" in config.feature_space.extractors

    def test_pipeline_not_dict_raises_error(self, tmp_path):
        """Test that pipeline as non-dict raises error."""
        # Arrange: Config with pipeline as list
        config_data = {"pipeline": ["n_d_v_i", "s_a_v_i"]}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="must be a dictionary"):
            load_config_yaml(config_path)

    def test_extractor_spec_not_dict_raises_error(self, tmp_path):
        """Test that extractor spec as non-dict raises error."""
        # Arrange: Config with extractor spec as string
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            f.write('{"pipeline": {"n_d_v_i": "NDVIExtractor"}}')

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="must be a dictionary"):
            load_config_json(config_path)

    def test_params_not_dict_raises_error(self, tmp_path):
        """Test that params as non-dict raises error."""
        # Arrange: Config with params as list
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json_str = (
                '{"pipeline": {"n_d_v_i": '
                '{"extractor": "NDVIExtractor", "params": [1, 2, 3]}}}'
            )
            f.write(json_str)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="must be a dictionary"):
            load_config_json(config_path)

    def test_invalid_extractor_params_raises_error(self, tmp_path):
        """Test that invalid extractor parameters raise error."""
        # Arrange: Config with invalid params for extractor
        config_data = {
            "pipeline": {
                "n_d_v_i": {
                    "extractor": "NDVIExtractor",
                    "params": {"invalid_param": 123},
                }
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="Failed to instantiate"):
            load_config_yaml(config_path)


class TestRunnerConfiguration:
    """Test runner configuration loading."""

    def test_config_get_default_runner(self):
        """Test that Config.get_default_runner() returns SequentialRunner."""
        # Act: Get default runner
        runner = Config.get_default_runner()

        # Assert: Returns SequentialRunner instance
        assert isinstance(runner, SequentialRunner)

    def test_config_without_runner_defaults_to_sequential(self):
        """Test that Config created without runner uses SequentialRunner."""
        # Arrange: Create FeatureSpace
        from hyppo.extractor import NDVIExtractor

        fs = FeatureSpace.from_list([NDVIExtractor()])

        # Act: Create Config without providing runner
        config = Config(feature_space=fs)

        # Assert: Runner defaults to SequentialRunner
        assert isinstance(config.runner, SequentialRunner)

    def test_load_config_without_runner_defaults_to_sequential(self, tmp_path):
        """Test that missing runner section defaults to SequentialRunner."""
        # Arrange: Config without runner section
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: Runner defaults to Sequential
        assert isinstance(config.runner, SequentialRunner)

    def test_load_config_with_sequential_runner(self, tmp_path):
        """Test loading config with explicit sequential runner."""
        # Arrange: Config with sequential runner
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}},
            "runner": {"type": "sequential"},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: Sequential runner configured
        assert isinstance(config.runner, SequentialRunner)

    def test_load_config_with_local_runner(self, tmp_path):
        """Test loading config with local process runner."""
        # Arrange: Config with local runner
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}},
            "runner": {"type": "local", "params": {"num_workers": 4}},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: Local runner configured
        from hyppo.runner import LocalProcessRunner
        assert isinstance(config.runner, LocalProcessRunner)

    def test_load_config_with_dask_threads_runner(self, tmp_path):
        """Test loading config with Dask threads runner."""
        # Arrange: Config with dask-threads runner
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}},
            "runner": {"type": "dask-threads", "params": {"num_threads": 8}},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: Dask threads runner configured
        from hyppo.runner import DaskThreadsRunner
        assert isinstance(config.runner, DaskThreadsRunner)

    def test_load_config_with_dask_processes_runner(self, tmp_path):
        """Test loading config with Dask processes runner."""
        # Arrange: Config with dask-processes runner
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}},
            "runner": {
                "type": "dask-processes",
                "params": {"num_workers": 4, "threads_per_worker": 2},
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act: Load configuration
        config = load_config_yaml(config_path)

        # Assert: Dask processes runner configured
        from hyppo.runner import DaskProcessesRunner
        assert isinstance(config.runner, DaskProcessesRunner)

    def test_runner_missing_type_raises_error(self, tmp_path):
        """Test that runner config without type raises error."""
        # Arrange: Config with runner but no type
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}},
            "runner": {"params": {"num_workers": 4}},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="type.*missing"):
            load_config_yaml(config_path)

    def test_runner_unknown_type_raises_error(self, tmp_path):
        """Test that unknown runner type raises error."""
        # Arrange: Config with invalid runner type
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}},
            "runner": {"type": "unknown-runner"},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="Unknown runner type"):
            load_config_yaml(config_path)

    def test_runner_not_dict_raises_error(self, tmp_path):
        """Test that runner as non-dict raises error."""
        # Arrange: Config with runner as string
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}},
            "runner": "sequential",
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="must be a dictionary"):
            load_config_yaml(config_path)

    def test_runner_params_not_dict_raises_error(self, tmp_path):
        """Test that runner params as non-dict raises error."""
        # Arrange: Config with params as list
        config_data = {
            "pipeline": {"n_d_v_i": {"extractor": "NDVIExtractor"}},
            "runner": {"type": "local", "params": [1, 2, 3]},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="must be a dictionary"):
            load_config_yaml(config_path)
