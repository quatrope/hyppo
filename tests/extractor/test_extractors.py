"""
Tests for the extractor modules.
"""

import pytest
import numpy as np
from hyppo.extractor.base import Extractor


class TestExtractorBase:
    """Test cases for the base Extractor class."""

    def test_extractor_is_abstract(self):
        """Test that Extractor base class cannot be instantiated."""
        with pytest.raises(TypeError):
            Extractor()

    def test_feature_name_generation(self):
        """Test automatic feature name generation."""

        class TestFeatureExtractor(Extractor):
            def extract(self, data):
                return {}

        class SimpleExtractor(Extractor):
            def extract(self, data):
                return {}

        class MyCustomExtractor(Extractor):
            def extract(self, data):
                return {}

        assert TestFeatureExtractor.feature_name() == "test"
        assert SimpleExtractor.feature_name() == "simple"
        assert MyCustomExtractor.feature_name() == "my_custom"
