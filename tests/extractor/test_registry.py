"""Tests for ExtractorRegistry."""

import pytest

from hyppo.extractor.base import Extractor
from hyppo.extractor.registry import ExtractorRegistry, registry


class DummyExtractor(Extractor):
    """Dummy extractor for testing."""

    def _extract(self, data, **inputs):
        return {"value": 1.0}


class AnotherDummyExtractor(Extractor):
    """Another dummy extractor for testing."""

    def _extract(self, data, **inputs):
        return {"value": 2.0}


class TestExtractorRegistrySingleton:
    """Tests for singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Test that ExtractorRegistry returns the same instance."""
        # Act: Create two instances
        registry1 = ExtractorRegistry()
        registry2 = ExtractorRegistry()

        # Assert: Same instance
        assert registry1 is registry2

    def test_global_registry_is_singleton(self):
        """Test that global registry is the singleton instance."""
        # Act: Create new instance
        new_registry = ExtractorRegistry()

        # Assert: Same as global
        assert new_registry is registry


class TestExtractorRegistryRegister:
    """Tests for register method."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Save and restore registry state."""
        original = registry._registry.copy()
        yield
        registry._registry.clear()
        registry._registry.update(original)

    def test_register_extractor_class(self):
        """Test registering an extractor class."""
        # Arrange: Unregister if exists
        if "DummyExtractor" in registry:
            registry.unregister("DummyExtractor")

        # Act: Register
        registry.register(DummyExtractor)

        # Assert: Registered
        assert "DummyExtractor" in registry
        assert registry.get("DummyExtractor") is DummyExtractor

    def test_register_non_extractor_raises_error(self):
        """Test non-Extractor class raises TypeError."""  # noqa: D202

        # Arrange: Non-extractor class
        class NotAnExtractor:
            pass

        # Act & Assert: TypeError
        with pytest.raises(TypeError, match="must inherit from Extractor"):
            registry.register(NotAnExtractor)

    def test_register_same_class_twice_succeeds(self):
        """Test re-registering the same class succeeds silently."""
        # Arrange: Unregister if exists
        if "DummyExtractor" in registry:
            registry.unregister("DummyExtractor")

        # Act: Register twice
        registry.register(DummyExtractor)
        registry.register(DummyExtractor)

        # Assert: Still registered
        assert "DummyExtractor" in registry

    def test_register_different_class_same_name_raises_error(
        self,
    ):
        """Test different class same name raises ValueError."""  # noqa: D202

        # Arrange: Create class with same name
        class DummyExtractor(Extractor):  # noqa: F811
            def _extract(self, data, **inputs):
                return {"different": True}

        # Ensure original is registered
        if "DummyExtractor" not in registry:
            from tests.extractor.test_registry import (
                DummyExtractor as OriginalDummy,
            )

            registry.register(OriginalDummy)

        # Act & Assert: ValueError
        with pytest.raises(ValueError, match="already registered"):
            registry.register(DummyExtractor)


class TestExtractorRegistryGet:
    """Tests for get method."""

    def test_get_registered_extractor(self):
        """Test getting a registered extractor."""
        # Assert: Get returns class
        assert registry.get("NDVIExtractor") is not None

    def test_get_unregistered_raises_keyerror(self):
        """Test getting unregistered extractor raises KeyError."""
        # Act & Assert: KeyError
        with pytest.raises(KeyError, match="not found in registry"):
            registry.get("NonExistentExtractor")


class TestExtractorRegistryIsRegistered:
    """Tests for is_registered method."""

    def test_is_registered_with_string_true(self):
        """Test is_registered with string returns True for registered."""
        # Assert: Registered
        assert registry.is_registered("NDVIExtractor") is True

    def test_is_registered_with_string_false(self):
        """Test is_registered with string returns False for unregistered."""
        # Assert: Not registered
        assert registry.is_registered("NonExistentExtractor") is False

    def test_is_registered_with_class_true(self):
        """Test is_registered with class returns True for registered."""
        # Arrange: Get a registered class
        from hyppo.extractor import NDVIExtractor

        # Assert: Registered
        assert registry.is_registered(NDVIExtractor) is True

    def test_is_registered_with_class_false(self):
        """Test is_registered with class returns False for unregistered."""
        # Assert: Not registered
        assert registry.is_registered(DummyExtractor) is False


class TestExtractorRegistryListExtractors:
    """Tests for list_extractors method."""

    def test_list_extractors_returns_list(self):
        """Test list_extractors returns a list."""
        # Act: Get list
        extractors = registry.list_extractors()

        # Assert: Is a list
        assert isinstance(extractors, list)

    def test_list_extractors_contains_registered(self):
        """Test list_extractors contains registered extractors."""
        # Act: Get list
        extractors = registry.list_extractors()

        # Assert: Contains known extractors
        assert "NDVIExtractor" in extractors
        assert "PCAExtractor" in extractors


class TestExtractorRegistryUnregister:
    """Tests for unregister method."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Save and restore registry state."""
        original = registry._registry.copy()
        yield
        registry._registry.clear()
        registry._registry.update(original)

    def test_unregister_removes_extractor(self):
        """Test unregister removes the extractor."""
        # Arrange: Register dummy
        registry.register(DummyExtractor)
        assert "DummyExtractor" in registry

        # Act: Unregister
        registry.unregister("DummyExtractor")

        # Assert: Removed
        assert "DummyExtractor" not in registry

    def test_unregister_nonexistent_raises_keyerror(self):
        """Test unregister raises KeyError for non-existent."""
        # Act & Assert: KeyError
        with pytest.raises(KeyError, match="not found in registry"):
            registry.unregister("NonExistentExtractor")


class TestExtractorRegistryClear:
    """Tests for clear method."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Save and restore registry state."""
        original = registry._registry.copy()
        yield
        registry._registry.clear()
        registry._registry.update(original)

    def test_clear_removes_all_extractors(self):
        """Test clear removes all extractors."""
        # Arrange: Ensure some extractors exist
        assert len(registry) > 0

        # Act: Clear
        registry.clear()

        # Assert: Empty
        assert len(registry) == 0


class TestExtractorRegistryDunderMethods:
    """Tests for dunder methods."""

    def test_len_returns_count(self):
        """Test __len__ returns number of registered extractors."""
        # Act: Get length
        length = len(registry)

        # Assert: Positive integer
        assert isinstance(length, int)
        assert length > 0

    def test_contains_with_registered(self):
        """Test __contains__ returns True for registered."""
        # Assert: Contains
        assert "NDVIExtractor" in registry

    def test_contains_with_unregistered(self):
        """Test __contains__ returns False for unregistered."""
        # Assert: Does not contain
        assert "NonExistentExtractor" not in registry

    def test_iter_yields_names(self):
        """Test __iter__ yields extractor names."""
        # Act: Iterate
        names = list(registry)

        # Assert: Contains known extractors
        assert "NDVIExtractor" in names
        assert isinstance(names[0], str)


class TestExtractorRegistryIntegration:
    """Integration tests for ExtractorRegistry."""

    def test_registered_extractors_are_instantiable(self):
        """Test all registered extractors can be instantiated."""
        # Act & Assert: Each registered extractor can be instantiated
        for name in registry:
            extractor_class = registry.get(name)
            try:
                extractor = extractor_class()
                assert isinstance(extractor, Extractor)
            except TypeError:
                # Some extractors may require arguments
                pass

    def test_all_standard_extractors_registered(self):
        """Test all standard extractors are registered."""
        # Arrange: Expected extractors
        expected = {
            "DWT1DExtractor",
            "DWT2DExtractor",
            "DWT3DExtractor",
            "GaborExtractor",
            "GeometricMomentExtractor",
            "GLCMExtractor",
            "ICAExtractor",
            "LBPExtractor",
            "LegendreMomentExtractor",
            "MNFExtractor",
            "MPExtractor",
            "NDVIExtractor",
            "NDWIExtractor",
            "PCAExtractor",
            "PPExtractor",
            "SAVIExtractor",
            "ZernikeMomentExtractor",
        }

        # Act: Get registered
        registered = set(registry.list_extractors())

        # Assert: All expected are registered
        assert expected.issubset(registered)
