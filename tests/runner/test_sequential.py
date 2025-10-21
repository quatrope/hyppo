"""Test cases for SequentialRunner."""

from hyppo.core import Feature, FeatureCollection, FeatureSpace, HSI
from hyppo.extractor.base import Extractor
from hyppo.runner import SequentialRunner
import numpy as np
import pytest


class SimpleExtractor(Extractor):
    """Simple extractor for testing."""

    def _extract(self, data: HSI, **inputs) -> dict:
        return {"value": np.mean(data.reflectance)}


class DependentExtractor(Extractor):
    """Extractor with dependencies for testing."""

    @classmethod
    def get_input_dependencies(cls):
        """Get input dependencies."""
        return {"input_data": {"extractor": SimpleExtractor, "required": True}}

    def _extract(self, data: HSI, **inputs) -> dict:
        input_value = inputs["input_data"]["value"]
        return {"result": input_value * 2}


class OptionalDependencyExtractor(Extractor):
    """Extractor with optional dependency."""

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Get input dependencies."""
        return {"optional_input": {"extractor": SimpleExtractor, "required": False}}

    @classmethod
    def get_input_default(cls, input_name: str):
        """Get input default."""
        if input_name == "optional_input":
            return SimpleExtractor()
        return None

    def _extract(self, data: HSI, **inputs) -> dict:
        if "optional_input" in inputs:
            return {"has_input": True, "value": inputs["optional_input"]["value"]}
        return {"has_input": False}


class CounterExtractor(Extractor):
    """Extractor that tracks execution order."""

    execution_count = 0
    execution_order = []

    def _extract(self, data: HSI, **inputs) -> dict:
        CounterExtractor.execution_count += 1
        order = CounterExtractor.execution_count
        CounterExtractor.execution_order.append(self.feature_name())
        return {"order": order}

    @classmethod
    def reset(cls):
        """Reset execution count and order."""
        cls.execution_count = 0
        cls.execution_order = []


class TestSequentialRunner:
    """Test cases for SequentialRunner."""

    def test_can_instantiate(self):
        """Test that SequentialRunner can be instantiated."""
        # Act: Create runner
        runner = SequentialRunner()

        # Assert: Instance created
        assert isinstance(runner, SequentialRunner)

    def test_resolve_single_extractor(self, small_hsi):
        """Test resolving single extractor without dependencies."""
        # Arrange: Create runner and feature space
        runner = SequentialRunner()
        extractor = SimpleExtractor()
        fs = FeatureSpace.from_list([extractor])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Verify results
        assert isinstance(results, FeatureCollection)
        assert len(results) == 1
        assert "simple" in results
        assert "value" in results["simple"]["data"]

    def test_resolve_multiple_extractors(self, small_hsi):
        """Test resolving multiple independent extractors."""
        # Act & Assert: Should raise error for duplicate extractors
        with pytest.raises(ValueError, match="Duplicate extractor"):
            FeatureSpace.from_list([SimpleExtractor(), SimpleExtractor()])

    def test_resolve_with_dependencies(self, small_hsi):
        """Test resolving extractors with dependencies."""
        # Arrange: Create runner with dependent extractors
        runner = SequentialRunner()
        simple = SimpleExtractor()
        dependent = DependentExtractor()
        fs = FeatureSpace.from_list([simple, dependent])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Verify both extractors executed and dependency passed
        assert len(results) == 2
        assert "simple" in results
        assert "dependent" in results
        assert "result" in results["dependent"]["data"]
        assert (
            results["dependent"]["data"]["result"]
            == results["simple"]["data"]["value"] * 2
        )

    def test_resolve_execution_order(self, small_hsi):
        """Test that extractors execute in correct topological order."""
        # Arrange: Reset counter and create feature space
        CounterExtractor.reset()

        class FirstCounter(CounterExtractor):
            pass

        class SecondCounter(CounterExtractor):
            @classmethod
            def get_input_dependencies(cls) -> dict:
                return {"input": {"extractor": FirstCounter, "required": True}}

        runner = SequentialRunner()
        fs = FeatureSpace.from_list([SecondCounter(), FirstCounter()])

        # Act: Execute extraction
        runner.resolve(small_hsi, fs)

        # Assert: First extractor executed before second
        assert CounterExtractor.execution_order == ["first_counter", "second_counter"]

    def test_resolve_with_optional_dependency_provided(self, small_hsi):
        """Test optional dependency when source is provided."""
        # Arrange: Create runner with optional dependency and source
        runner = SequentialRunner()
        simple = SimpleExtractor()
        optional = OptionalDependencyExtractor()
        fs = FeatureSpace.from_list([simple, optional])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Optional input was used
        assert results["optional_dependency"]["data"]["has_input"] is True
        assert "value" in results["optional_dependency"]["data"]

    def test_resolve_with_optional_dependency_default(self, small_hsi):
        """Test optional dependency using default when source not provided."""
        # Arrange: Create runner with only optional dependency extractor
        runner = SequentialRunner()
        optional = OptionalDependencyExtractor()
        fs = FeatureSpace.from_list([optional])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Default was used
        assert len(results) == 1
        assert "optional_dependency" in results
        assert results["optional_dependency"]["data"]["has_input"] is True

    def test_get_defaults_for_extractor_no_defaults(self):
        """Test _get_defaults_for_extractor with no optional inputs."""
        # Arrange: Create runner and extractor without optional inputs
        runner = SequentialRunner()
        extractor = SimpleExtractor()

        # Act: Get defaults
        defaults = runner._get_defaults_for_extractor(extractor)

        # Assert: No defaults returned
        assert defaults == {}

    def test_get_defaults_for_extractor_with_optional(self):
        """Test _get_defaults_for_extractor with optional inputs."""
        # Arrange: Create runner and extractor with optional input
        runner = SequentialRunner()
        extractor = OptionalDependencyExtractor()

        # Act: Get defaults
        defaults = runner._get_defaults_for_extractor(extractor)

        # Assert: Default extractor returned
        assert "optional_input" in defaults
        assert isinstance(defaults["optional_input"], SimpleExtractor)

    def test_get_defaults_for_extractor_required_dependency(self):
        """Test _get_defaults_for_extractor ignores required dependencies."""
        # Arrange: Create runner and extractor with required dependency
        runner = SequentialRunner()
        extractor = DependentExtractor()

        # Act: Get defaults
        defaults = runner._get_defaults_for_extractor(extractor)

        # Assert: Required dependency not in defaults
        assert "input_data" not in defaults
        assert defaults == {}

    def test_resolve_empty_feature_space(self, small_hsi):
        """Test resolving with empty feature space."""
        # Arrange: Create empty feature space
        runner = SequentialRunner()
        fs = FeatureSpace({})

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Empty results
        assert len(results) == 0

    def test_integration_with_real_extractors(self, small_hsi):
        """Test SequentialRunner with real extractor implementations."""
        # Arrange: Import real extractors
        from hyppo.extractor import MeanExtractor, StdExtractor

        runner = SequentialRunner()
        fs = FeatureSpace.from_list([MeanExtractor(), StdExtractor()])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Both extractors produced results
        assert len(results) == 2
        assert "mean" in results
        assert "std" in results
        assert isinstance(results["mean"], Feature)
        assert isinstance(results["std"], Feature)
