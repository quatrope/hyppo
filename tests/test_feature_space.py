"""
Tests for the FeatureSpace module.
"""

import pytest
from hyppo.feature_space import FeatureSpace
from hyppo.extractor.base import Extractor


class MockExtractor(Extractor):
    """Mock extractor for testing purposes."""

    def extract(self, data):
        return {"mock_feature": 42.0}


def test_feature_space_initialization():
    """Test FeatureSpace initialization with extractors."""
    extractor = MockExtractor()
    extractors = {"mock": extractor}

    fs = FeatureSpace(extractors)

    assert len(fs.get_extractors()) == 1
    assert "mock" in fs.get_extractors()
    assert isinstance(fs.get_extractors()["mock"], MockExtractor)


def test_feature_space_empty_extractors():
    """Test that FeatureSpace raises error with empty extractors."""
    with pytest.raises(ValueError, match="No extractors supplied"):
        FeatureSpace({})


def test_feature_space_invalid_extractor():
    """Test that FeatureSpace raises error with invalid extractor."""
    with pytest.raises(TypeError, match="must be an Extractor"):
        FeatureSpace({"invalid": "not_an_extractor"})


def test_feature_space_from_features():
    """Test FeatureSpace.from_features() class method."""
    # This test will need to be updated based on available extractors
    # For now, we'll test the error case for unknown features
    with pytest.raises(ValueError, match="Feature desconocida"):
        FeatureSpace.from_features(["nonexistent_feature"])


def test_feature_space_extract_with_mock(small_hsi):
    """Test feature extraction with mock extractor."""
    extractor = MockExtractor()
    extractors = {"mock": extractor}

    fs = FeatureSpace(extractors)

    # Test with default runner (should create ThreadsRunner)
    result = fs.extract(small_hsi)

    # The exact result format depends on the runner implementation
    assert result is not None
