import pytest
import tempfile
import yaml
import json
from pathlib import Path

from hyppo.io import parse_config, ConfigExecutor
from hyppo.core import FeatureSpace
from hyppo import io
from tests.fixtures.hsi import sample_hsi


@pytest.mark.skip(reason="Not yet properly implemented")
class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_yaml_config_to_feature_space(self, tmp_path):
        """Test complete workflow from YAML config to FeatureSpace."""
        # Create test config
        config_data = {
            "input": "test.h5",
            "pipeline": {
                "mean": {"extractor": "MeanExtractor"},
                "std": {"extractor": "StdExtractor"},
            },
            "output": ["mean", "std"],
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Parse config
        config = parse_config(config_path)

        # Build FeatureSpace
        executor = ConfigExecutor(config, validate=False)  # Skip validation for test
        feature_space = executor.build_feature_space()

        assert isinstance(feature_space, FeatureSpace)
        assert len(feature_space.extractors) == 2
        assert "mean" in feature_space.extractors
        assert "std" in feature_space.extractors

    def test_json_config_to_feature_space(self, tmp_path):
        """Test complete workflow from JSON config to FeatureSpace."""
        config_data = {
            "input": "test.h5",
            "pipeline": {"median": {"extractor": "MedianExtractor"}},
        }

        config_path = tmp_path / "test_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Parse and build
        config = parse_config(config_path)
        executor = ConfigExecutor(config, validate=False)
        feature_space = executor.build_feature_space()

        assert isinstance(feature_space, FeatureSpace)
        assert "median" in feature_space.extractors

    def test_feature_space_from_config_classmethod(self, tmp_path):
        """Test FeatureSpace.from_config() class method."""
        config_data = {
            "input": "test.h5",
            "pipeline": {"max": {"extractor": "MaxExtractor"}},
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Use class method
        feature_space = FeatureSpace.from_config(config_path, validate=False)

        assert isinstance(feature_space, FeatureSpace)
        assert "max" in feature_space.extractors

    def test_config_with_dependencies(self, tmp_path):
        """Test configuration with extractor dependencies."""
        config_data = {
            "input": "test.h5",
            "pipeline": {
                "mean": {"extractor": "MeanExtractor"},
                "gabor": {"extractor": "GaborExtractor"},
            },
        }

        config_path = tmp_path / "complex_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        feature_space = FeatureSpace.from_config(config_path, validate=False)

        # Check dependency graph is built properly
        execution_order = feature_space.feature_graph.get_execution_order()
        assert len(execution_order) == 2
        assert "mean" in execution_order
        assert "gabor" in execution_order

    def test_config_execution_with_mock_hsi(self, sample_hsi, tmp_path):
        """Test complete execution with mock HSI data."""
        # Create a simple config
        config_data = {
            "input": "dummy.h5",  # Won't be used in this test
            "pipeline": {
                "mean": {"extractor": "MeanExtractor"},
                "std": {"extractor": "StdExtractor"},
            },
            "output": ["mean"],
        }

        config_path = tmp_path / "exec_test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Build FeatureSpace and execute
        feature_space = FeatureSpace.from_config(config_path, validate=False)
        results = feature_space.extract(sample_hsi)

        # Check results structure
        assert len(results) == 2  # Both extractors run despite output filter
        assert "mean" in results
        assert "std" in results

        # Check that results have the expected structure
        mean_result = results["mean"]
        assert hasattr(mean_result, "data")
        assert "mean" in mean_result.data

    @pytest.mark.parametrize("config_format", ["yaml", "json"])
    def test_config_format_consistency(self, config_format, tmp_path):
        """Test that YAML and JSON configs produce identical results."""
        config_data = {
            "input": "test.h5",
            "pipeline": {"min": {"extractor": "MinExtractor"}},
        }

        if config_format == "yaml":
            config_path = tmp_path / "test.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
        else:
            config_path = tmp_path / "test.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f)

        feature_space = FeatureSpace.from_config(config_path, validate=False)

        assert isinstance(feature_space, FeatureSpace)
        assert "min" in feature_space.extractors
        assert feature_space.extractors["min"].__class__.__name__ == "MinExtractor"
