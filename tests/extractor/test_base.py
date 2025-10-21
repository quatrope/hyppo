from hyppo.core import HSI
from hyppo.extractor.base import Extractor
import pytest


class ConcreteExtractor(Extractor):
    """Concrete implementation for testing purposes."""

    def _extract(self, data: HSI, **inputs) -> dict:
        return {"feature": "value"}


class ConcreteExtractorWithValidation(Extractor):
    """Concrete implementation with custom validation."""

    def _validate(self, data: HSI, **inputs):
        if data is None:
            raise ValueError("Data cannot be None")

    def _extract(self, data: HSI, **inputs) -> dict:
        return {"validated_feature": "value"}


class ConcreteExtractorWithDependencies(Extractor):
    """Concrete implementation with input dependencies."""

    @classmethod
    def get_input_dependencies(cls) -> dict:
        return {"mean": "mean", "std": "std"}

    def _extract(self, data: HSI, **inputs) -> dict:
        return {"result": inputs}


class ConcreteExtractorWithDefaults(Extractor):
    """Concrete implementation with default inputs."""

    @classmethod
    def get_input_default(cls, input_name: str) -> "Extractor | None":
        if input_name == "mean":
            return ConcreteExtractor()
        return None

    def _extract(self, data: HSI, **inputs) -> dict:
        return {"with_defaults": True}


class MyFeatureExtractor(Extractor):
    """Test feature name generation with 'FeatureExtractor' suffix."""

    def _extract(self, data: HSI, **inputs) -> dict:
        return {}


class SimpleExtractor(Extractor):
    """Test feature name generation with 'Extractor' suffix."""

    def _extract(self, data: HSI, **inputs) -> dict:
        return {}


class GLCMExtractor(Extractor):
    """Test feature name generation with acronym."""

    def _extract(self, data: HSI, **inputs) -> dict:
        return {}


class NoSuffix(Extractor):
    """Test feature name generation without common suffixes."""

    def _extract(self, data: HSI, **inputs) -> dict:
        return {}


class TestExtractor:
    """Test cases for the base Extractor class."""

    def test_extractor_is_abstract(self):
        """Test that Extractor base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Extractor()  # type: ignore

    def test_basic_extract_call(self, small_hsi):
        """Test extract() happy path."""
        # Arrange: Create test extractor
        extractor = ConcreteExtractor()

        # Act: Call extract method
        result = extractor.extract(small_hsi)

        # Assert: Verify expected output
        assert result == {"feature": "value"}

    def test_validate_is_called_before_extract(self):
        """Test that _validate() is executed before _extract() and can raise errors."""
        # Arrange: Create extractor with validation logic
        extractor = ConcreteExtractorWithValidation()

        # Act & Assert: Verify validation raises error for invalid data
        with pytest.raises(ValueError, match="Data cannot be None"):
            extractor.extract(None)  # type: ignore

    def test_validate_passes_with_valid_data(self, small_hsi):
        """Test that _validate() allows execution to proceed with valid data."""
        # Arrange: Create extractor
        extractor = ConcreteExtractorWithValidation()

        # Act: Extract with valid data
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert result == {"validated_feature": "value"}

    def test_get_input_dependencies_default(self):
        """Test that get_input_dependencies() returns empty dict by default."""
        # Arrange & Act: Get default dependencies
        dependencies = ConcreteExtractor.get_input_dependencies()

        # Assert: Verify empty dict returned
        assert dependencies == {}

    def test_get_input_dependencies_custom(self):
        """Test that get_input_dependencies() can be overridden to return custom dependencies."""
        # Arrange & Act: Get custom dependencies from overridden method
        dependencies = ConcreteExtractorWithDependencies.get_input_dependencies()

        # Assert: Verify custom dependencies returned
        assert dependencies == {"mean": "mean", "std": "std"}

    def test_get_input_default_returns_none(self):
        """Test that get_input_default() returns None by default."""
        # Arrange & Act: Get default for arbitrary input name
        default = ConcreteExtractor.get_input_default("some_input")

        # Assert: Verify None returned
        assert default is None

    def test_get_input_default_custom(self):
        """Test that get_input_default() can be overridden to return extractor instances."""
        # Arrange & Act: Get defaults for known and unknown inputs
        default_mean = ConcreteExtractorWithDefaults.get_input_default("mean")
        default_other = ConcreteExtractorWithDefaults.get_input_default("other")

        # Assert: Verify correct defaults returned
        assert isinstance(default_mean, ConcreteExtractor)
        assert default_other is None

    def test_feature_name_with_feature_extractor_suffix(self):
        """Test feature_name() removes 'FeatureExtractor' suffix and converts to snake_case."""
        # Arrange & Act: Generate name from class with FeatureExtractor suffix
        name = MyFeatureExtractor.feature_name()

        # Assert: Verify suffix removed and converted to snake_case
        assert name == "my"

    def test_feature_name_with_extractor_suffix(self):
        """Test feature_name() removes 'Extractor' suffix and converts to snake_case."""
        # Arrange & Act: Generate name from class with Extractor suffix
        name = SimpleExtractor.feature_name()

        # Assert: Verify suffix removed and converted to snake_case
        assert name == "simple"

    def test_feature_name_with_acronym(self):
        """Test feature_name() splits acronyms into separate words."""
        # Arrange & Act: Generate name from acronym class
        name = GLCMExtractor.feature_name()

        # Assert: Verify acronym split into individual letters
        assert name == "g_l_c_m"

    def test_feature_name_without_suffix(self):
        """Test feature_name() converts class names without common suffixes."""
        # Arrange & Act: Generate name from class without standard suffixes
        name = NoSuffix.feature_name()

        # Assert: Verify CamelCase converted to snake_case
        assert name == "no_suffix"

    def test_feature_name_camel_case_conversion(self):
        """Test feature_name() correctly splits CamelCase into snake_case."""

        # Arrange: Create class with complex CamelCase name
        class MyComplexFeatureName(Extractor):
            def _extract(self, data: HSI, **inputs) -> dict:
                return {}

        # Act: Generate feature name
        name = MyComplexFeatureName.feature_name()

        # Assert: Verify all words split and converted to snake_case
        assert name == "my_complex_feature_name"
