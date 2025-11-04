"""Tests for BaseRunner."""

import pytest

from hyppo.core import FeatureSpace, HSI
from hyppo.runner import BaseRunner


class ConcreteRunner(BaseRunner):
    """Concrete implementation for testing purposes."""

    def resolve(self, data: HSI, feature_space) -> dict:
        """Execute test resolution."""
        return {"test": "result"}


class TestBaseRunner:
    """Test cases for the BaseRunner abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseRunner cannot be instantiated directly."""
        # Act & Assert: Attempt to instantiate abstract class
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class"
        ):
            BaseRunner()  # type: ignore

    def test_concrete_subclass_can_be_instantiated(self):
        """Test that concrete subclass can be instantiated."""
        # Act: Instantiate concrete implementation
        runner = ConcreteRunner()

        # Assert: Instance is created successfully
        assert isinstance(runner, BaseRunner)
        assert isinstance(runner, ConcreteRunner)

    def test_resolve_is_abstract(self):
        """Test that resolve method is abstract."""
        # Arrange: Get abstract methods
        abstract_methods = BaseRunner.__abstractmethods__

        # Assert: resolve is in abstract methods
        assert "resolve" in abstract_methods

    def test_resolve_signature(self, small_hsi):
        """Test that resolve method has correct signature."""
        # Arrange: Create concrete runner and mock feature space
        runner = ConcreteRunner()

        feature_space = FeatureSpace({})

        # Act: Call resolve
        result = runner.resolve(small_hsi, feature_space)

        # Assert: Returns dict
        assert isinstance(result, dict)

    def test_subclass_without_resolve_fails(self):
        """Test subclass without resolve cannot be instantiated."""
        # Arrange: Define incomplete subclass
        class IncompleteRunner(BaseRunner):
            pass

        # Act & Assert: Attempt to instantiate
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class"
        ):
            IncompleteRunner()  # type: ignore
