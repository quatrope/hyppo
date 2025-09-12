import pytest
from hyppo.extractor.base import InputDependency, Extractor
from hyppo.extractor.mean import MeanExtractor
from hyppo.extractor.mnf import MNFExtractor


class TestInputDependency:
    """Test the InputDependency class."""

    def test_input_dependency_creation(self):
        """Test creating an InputDependency."""
        dep = InputDependency(
            name="test_input", extractor_type=MeanExtractor, required=True
        )

        assert dep.name == "test_input"
        assert dep.extractor_type == MeanExtractor
        assert dep.required is True
        assert dep.default_config is None

    def test_input_dependency_with_defaults(self):
        """Test InputDependency with default configuration."""
        dep = InputDependency(
            name="optional_input",
            extractor_type=MNFExtractor,
            required=False,
            default_config={"n_components": 5, "whiten": False},
        )

        assert dep.name == "optional_input"
        assert dep.required is False
        assert dep.default_config == {"n_components": 5, "whiten": False}


class TestExtractorBase:
    """Test cases for the base Extractor class."""

    def test_extractor_is_abstract(self):
        """Test that Extractor base class cannot be instantiated."""
        with pytest.raises(TypeError):
            Extractor()
