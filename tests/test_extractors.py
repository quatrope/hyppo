"""
Tests for the extractor modules.
"""

import pytest
import numpy as np
from hyppo.extractor.base import Extractor
from hyppo.utils import get_all_extractors


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


class TestExtractorRegistry:
    """Test cases for extractor registry functionality."""

    def test_get_all_extractors(self):
        """Test that get_all_extractors returns extractor classes."""
        extractors = get_all_extractors()

        assert isinstance(extractors, list)
        assert len(extractors) > 0

        for extractor_cls in extractors:
            assert issubclass(extractor_cls, Extractor)
            assert hasattr(extractor_cls, "extract")
            assert hasattr(extractor_cls, "feature_name")

    def test_extractor_names_unique(self):
        """Test that extractor feature names are unique."""
        extractors = get_all_extractors()
        feature_names = [ext.feature_name() for ext in extractors]

        assert len(feature_names) == len(
            set(feature_names)
        ), f"Duplicate feature names found: {feature_names}"


class TestConcreteExtractors:
    """Test cases for concrete extractor implementations."""

    @pytest.mark.parametrize("extractor_cls", get_all_extractors())
    def test_extractor_instantiation(self, extractor_cls):
        """Test that all registered extractors can be instantiated."""
        try:
            if extractor_cls.__name__ == "DummyExtractor":
                extractor = extractor_cls(result=42)
            else:
                extractor = extractor_cls()
            assert isinstance(extractor, Extractor)
        except Exception as e:
            pytest.fail(f"Failed to instantiate {extractor_cls.__name__}: {e}")

    @pytest.mark.parametrize("extractor_cls", get_all_extractors())
    def test_extractor_extract_method(self, extractor_cls, small_hsi):
        """Test that all extractors have working extract methods."""
        try:
            if extractor_cls.__name__ == "DummyExtractor":
                extractor = extractor_cls(result=42)
            else:
                extractor = extractor_cls()
            result = extractor.extract(small_hsi)

            assert isinstance(
                result, dict
            ), f"{extractor_cls.__name__}.extract() must return a dict"
            assert (
                len(result) > 0
            ), f"{extractor_cls.__name__}.extract() returned empty dict"

            # Check if this is a simple extractor (like DummyExtractor) or complex one
            if "features" in result:
                # Complex extractors return features and wavelengths
                assert "wavelengths" in result, "Complex extractors must return wavelengths"
                assert isinstance(result["features"], np.ndarray), "Features must be numpy array"
                assert isinstance(result["wavelengths"], np.ndarray), "Wavelengths must be numpy array"
            else:
                # Simple extractors return direct key-value pairs
                for key, value in result.items():
                    assert isinstance(key, str), f"Feature name must be string, got {type(key)}"
                    assert isinstance(value, (int, float, np.number)), f"Feature value must be numeric, got {type(value)}"

        except NotImplementedError:
            pytest.skip(f"{extractor_cls.__name__}.extract() not implemented yet")
        except Exception as e:
            pytest.fail(f"{extractor_cls.__name__}.extract() failed: {e}")

    @pytest.mark.parametrize("extractor_cls", get_all_extractors())
    def test_extractor_validate_method(self, extractor_cls):
        """Test that all extractors have validate methods."""
        try:
            if extractor_cls.__name__ == "DummyExtractor":
                extractor = extractor_cls(result=42)
            else:
                extractor = extractor_cls()
            # validate() should not raise an exception for default params
            extractor.validate()
        except NotImplementedError:
            pytest.skip(f"{extractor_cls.__name__}.validate() not implemented yet")
        except Exception as e:
            pytest.fail(f"{extractor_cls.__name__}.validate() failed: {e}")
