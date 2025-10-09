import pytest
from hyppo.extractor.std import StdExtractor
from hyppo.core import FeatureSpace
from tests.fixtures.extractors import SimpleExtractor, MediumExtractor
import hyppo


class TestFeatureSpace:
    """Tests for FeatureSpace."""

    def test_basic_pipeline(self, sample_hsi):
        """Test basic pipeline with real extractors."""
        pipeline_config = {
            "mean": (hyppo.extractor.MeanExtractor(), {}),
            "std": (hyppo.extractor.StdExtractor(), {}),
            "min_val": (hyppo.extractor.MinExtractor(), {}),
        }

        fs = FeatureSpace(pipeline_config)
        results = fs.extract(sample_hsi)

        # Check that all extractors ran
        assert len(results) == 3
        assert "mean" in results
        assert "std" in results
        assert "min_val" in results

        # Check that results have expected structure
        for result in results.values():
            assert hasattr(result, "data")
            assert isinstance(result.data, dict)

    def test_feature_space_creation(self):
        """Test creating FeatureSpace with typed configuration."""
        pipeline_config = {
            "simple": (SimpleExtractor(), {}),
            "medium": (MediumExtractor(), {"simple_input": "simple"}),
        }

        fs = FeatureSpace(pipeline_config)

        assert fs.feature_graph is not None

        # Test validation passed
        assert len(fs.extractors) == 2

    def test_feature_space_invalid_extractor_type(self):
        """Test FeatureSpace with invalid expected types for inputs configuration."""
        # Wrong type mapping
        pipeline_config = {
            "simple": (StdExtractor(), {}),  # std as simple
            "medium": (MediumExtractor(), {"simple_input": "simple"}),
        }

        with pytest.raises(TypeError):
            FeatureSpace(pipeline_config)

    def test_from_list_simple(self, sample_hsi):
        """Test creating FeatureSpace from a simple list without dependencies."""
        extractors = [
            hyppo.extractor.MeanExtractor(),
            hyppo.extractor.StdExtractor(),
            hyppo.extractor.MinExtractor(),
        ]

        fs = FeatureSpace.from_list(extractors)
        results = fs.extract(sample_hsi)

        # Check that all extractors ran
        assert len(results) == 3
        assert "mean" in results
        assert "std" in results
        assert "min" in results

    def test_from_list_with_dependencies(self, sample_hsi):
        """Test creating FeatureSpace from list with dependencies."""
        from tests.fixtures.extractors import SimpleExtractor, MediumExtractor

        extractors = [
            SimpleExtractor(),
            MediumExtractor(),
        ]

        fs = FeatureSpace.from_list(extractors)
        results = fs.extract(sample_hsi)

        # Check that both extractors ran with proper dependency resolution
        assert len(results) == 2
        assert "simple" in results
        assert "medium" in results

    def test_from_list_duplicate_types_error(self):
        """Test error when list contains duplicate extractor types."""
        extractors = [
            hyppo.extractor.MeanExtractor(),
            hyppo.extractor.MeanExtractor(),  # Duplicate
        ]

        with pytest.raises(ValueError, match="Duplicate extractor name"):
            FeatureSpace.from_list(extractors)

    def test_from_list_missing_dependency_error(self):
        """Test error when required dependency is missing."""
        from tests.fixtures.extractors import MediumExtractor

        extractors = [
            MediumExtractor(),  # Requires SimpleExtractor but it's missing
        ]

        with pytest.raises(ValueError, match="Missing required dependency"):
            FeatureSpace.from_list(extractors)

    def test_from_list_complex_chain(self, sample_hsi):
        """Test complex dependency chain resolution."""
        from tests.fixtures.extractors import (
            SimpleExtractor,
            MediumExtractor,
            AdvancedExtractor,
        )

        extractors = [
            SimpleExtractor(),  # no deps
            MediumExtractor(),  # deps: simple
            AdvancedExtractor(),  # deps: medium, simple1, simple2 (but simple2 is optional)
        ]

        fs = FeatureSpace.from_list(extractors)
        results = fs.extract(sample_hsi)

        # All should work, advanced should resolve its dependencies
        assert len(results) == 3
        assert "simple" in results
        assert "medium" in results
        assert "advanced" in results

    def test_from_list_empty_list(self):
        """Test creating FeatureSpace from empty list."""
        fs = FeatureSpace.from_list([])
        assert len(fs.extractors) == 0

    def test_from_list_required_dependency_missing_fails(self):
        """Test that missing required dependencies cause errors."""
        from tests.fixtures.extractors import AdvancedExtractor, SimpleExtractor

        # AdvancedExtractor requires MediumExtractor but we don't provide it
        extractors = [
            SimpleExtractor(),
            AdvancedExtractor(),
        ]

        # This should fail because MediumExtractor (medium_input) is required
        with pytest.raises(ValueError, match="Missing required dependency"):
            FeatureSpace.from_list(extractors)
