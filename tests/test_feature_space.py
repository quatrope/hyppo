import pytest
from hyppo.extractor.std import StdExtractor
from hyppo.feature_space import FeatureSpace
from tests.fixtures.extractors import SimpleExtractor, MediumExtractor


class TestFeatureSpace:
    """Tests for FeatureSpace."""

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
