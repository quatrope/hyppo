"""Tests for FeatureSpace."""

import pytest

import hyppo
from hyppo.core import FeatureSpace
from hyppo.extractor.std import StdExtractor
from tests.fixtures.extractors import MediumExtractor, SimpleExtractor


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
        """Test FeatureSpace with invalid types for inputs config."""
        # Wrong type mapping
        pipeline_config = {
            "simple": (StdExtractor(), {}),  # std as simple
            "medium": (MediumExtractor(), {"simple_input": "simple"}),
        }

        with pytest.raises(TypeError):
            FeatureSpace(pipeline_config)

    def test_from_list_simple(self, sample_hsi):
        """Test creating FeatureSpace from list without dependencies."""
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
        from tests.fixtures.extractors import MediumExtractor, SimpleExtractor

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
            AdvancedExtractor,
            MediumExtractor,
            SimpleExtractor,
        )

        extractors = [
            SimpleExtractor(),  # no deps
            MediumExtractor(),  # deps: simple
            # deps: medium, simple1, simple2 (simple2 is optional)
            AdvancedExtractor(),
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
        from tests.fixtures.extractors import (
            AdvancedExtractor,
            SimpleExtractor,
        )

        # AdvancedExtractor requires MediumExtractor but we don't provide it
        extractors = [
            SimpleExtractor(),
            AdvancedExtractor(),
        ]

        # This should fail because MediumExtractor (medium_input) is required
        with pytest.raises(ValueError, match="Missing required dependency"):
            FeatureSpace.from_list(extractors)

    def test_get_extractors(self):
        """Test get_extractors method."""
        pipeline_config = {
            "simple": (SimpleExtractor(), {}),
        }

        fs = FeatureSpace(pipeline_config)
        extractors = fs.get_extractors()

        assert len(extractors) == 1
        assert "simple" in extractors
        assert isinstance(extractors["simple"], SimpleExtractor)

    def test_from_list_ambiguous_dependency(self):
        """Test error when multiple extractors match same required type."""
        from tests.fixtures.extractors import MediumExtractor, SimpleExtractor

        # To test ambiguous dependency, we need two DIFFERENT extractors
        # that are both instances of the same base type. Use subclasses.

        class SimpleExtractorA(SimpleExtractor):
            pass

        class SimpleExtractorB(SimpleExtractor):
            pass

        # MediumExtractor requires SimpleExtractor, and both A and B
        # are instances of it
        extractors = [
            SimpleExtractorA(),
            SimpleExtractorB(),
            MediumExtractor(),
        ]

        # This should fail with ambiguous dependency
        with pytest.raises(ValueError, match="Ambiguous dependency"):
            FeatureSpace.from_list(extractors)

    def test_from_list_duplicate_exact_type(self):
        """Test duplicate type check catches same type added twice."""
        # This test tries to hit line 92, but it's likely unreachable
        # because duplicate types generate duplicate names which are
        # caught at line 85
        mean1 = hyppo.extractor.MeanExtractor()
        mean2 = hyppo.extractor.MeanExtractor()

        extractors = [mean1, mean2]

        # Will be caught by duplicate name check first
        with pytest.raises(ValueError, match="Duplicate"):
            FeatureSpace.from_list(extractors)
