"""Tests for Bunch."""

import copy

import pytest

from hyppo.utils.bunch import Bunch


class TestBunch:
    """Test cases for the Bunch container class."""

    def test_initialization_with_dict(self):
        """Test Bunch initialization with a dictionary."""
        # Arrange: Create data dictionary
        data = {"a": 1, "b": 2}

        # Act: Initialize Bunch
        bunch = Bunch("test", data)

        # Assert: Verify initialization
        assert bunch["a"] == 1
        assert bunch["b"] == 2

    def test_initialization_with_invalid_type(self):
        """Test Bunch initialization fails with non-mapping type."""
        # Arrange: Create non-mapping data
        data = [1, 2, 3]

        # Act & Assert: Verify TypeError raised
        with pytest.raises(TypeError, match="Data must be some kind of mapping"):
            Bunch("test", data)

    def test_getitem_access(self):
        """Test dictionary-style item access."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {"key": "value"})

        # Act: Access via getitem
        result = bunch["key"]

        # Assert: Verify correct value
        assert result == "value"

    def test_getattr_access(self):
        """Test attribute-style access."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {"attr": "value"})

        # Act: Access via attribute
        result = bunch.attr

        # Assert: Verify correct value
        assert result == "value"

    def test_getattr_raises_attribute_error(self):
        """Test attribute access raises AttributeError for missing keys."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {})

        # Act & Assert: Verify AttributeError raised
        with pytest.raises(AttributeError):
            _ = bunch.nonexistent

    def test_setattr_raises_attribute_error(self):
        """Test Bunch is read-only via attribute assignment."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {"key": "value"})

        # Act & Assert: Verify AttributeError raised
        with pytest.raises(AttributeError, match="Bunch 'test' is read-only"):
            bunch.key = "new_value"

    def test_iter(self):
        """Test iteration over Bunch keys."""
        # Arrange: Create Bunch
        data = {"a": 1, "b": 2, "c": 3}
        bunch = Bunch("test", data)

        # Act: Iterate over keys
        keys = list(bunch)

        # Assert: Verify all keys present
        assert set(keys) == {"a", "b", "c"}

    def test_len(self):
        """Test length of Bunch."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {"a": 1, "b": 2, "c": 3})

        # Act: Get length
        length = len(bunch)

        # Assert: Verify correct length
        assert length == 3

    def test_repr(self):
        """Test string representation of Bunch."""
        # Arrange: Create Bunch
        bunch = Bunch("test_name", {"a": 1, "b": 2})

        # Act: Get repr
        result = repr(bunch)

        # Assert: Verify format and content
        assert result.startswith("<test_name")
        assert "a" in result or "b" in result

    def test_repr_empty(self):
        """Test string representation of empty Bunch."""
        # Arrange: Create empty Bunch
        bunch = Bunch("empty", {})

        # Act: Get repr
        result = repr(bunch)

        # Assert: Verify empty representation
        assert result == "<empty {}>"

    def test_dir(self):
        """Test dir() includes both Bunch attributes and data keys."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {"custom_key": "value"})

        # Act: Get dir listing
        dir_result = dir(bunch)

        # Assert: Verify custom key present
        assert "custom_key" in dir_result

    def test_get_with_existing_key(self):
        """Test get() method with existing key."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {"key": "value"})

        # Act: Get value
        result = bunch.get("key")

        # Assert: Verify correct value
        assert result == "value"

    def test_get_with_missing_key(self):
        """Test get() method with missing key returns default."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {})

        # Act: Get with default
        result = bunch.get("missing", "default")

        # Assert: Verify default returned
        assert result == "default"

    def test_get_with_missing_key_no_default(self):
        """Test get() method with missing key and no default."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {})

        # Act: Get without default
        result = bunch.get("missing")

        # Assert: Verify None returned
        assert result is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Arrange: Create Bunch
        data = {"a": 1, "b": [2, 3], "c": {"nested": "value"}}
        bunch = Bunch("test", data)

        # Act: Convert to dict
        result = bunch.to_dict()

        # Assert: Verify deep copy
        assert result == data
        assert result is not data

    def test_copy(self):
        """Test shallow copy of Bunch."""
        # Arrange: Create Bunch
        data = {"a": 1, "b": [2, 3]}
        bunch = Bunch("test", data)

        # Act: Copy Bunch
        bunch_copy = copy.copy(bunch)

        # Assert: Verify copy characteristics
        assert bunch_copy["a"] == bunch["a"]
        assert bunch_copy._name == bunch._name
        assert bunch_copy._data is bunch._data

    def test_deepcopy(self):
        """Test deep copy of Bunch."""
        # Arrange: Create Bunch with nested data
        data = {"a": 1, "b": [2, 3], "c": {"nested": "value"}}
        bunch = Bunch("test", data)

        # Act: Deep copy Bunch
        bunch_copy = copy.deepcopy(bunch)

        # Assert: Verify deep copy characteristics
        assert bunch_copy["a"] == bunch["a"]
        assert bunch_copy._data is not bunch._data
        assert bunch_copy["b"] is not bunch["b"]
        assert bunch_copy["c"] is not bunch["c"]

    def test_setstate(self):
        """Test __setstate__ for pickle/multiprocessing support."""
        # Arrange: Create Bunch
        bunch = Bunch("test", {"key": "value"})
        state = {"_name": "new_name", "_data": {"new_key": "new_value"}}

        # Act: Set state
        bunch.__setstate__(state)

        # Assert: Verify state updated
        assert bunch._name == "new_name"
        assert bunch["new_key"] == "new_value"
