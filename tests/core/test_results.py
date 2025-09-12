import pytest
import numpy as np
from hyppo.core import FeatureResult, FeatureResultCollection
from hyppo.extractor.mean import MeanExtractor


class TestFeatureResult:
    """Tests for FeatureResult class."""

    def test_feature_result_creation(self):
        """Test creating FeatureResult from dictionary."""
        data = {"mean": [1, 2, 3], "std": [0.1, 0.2, 0.3]}
        result = FeatureResult(data)

        assert result == data
        assert isinstance(result, dict)

    def test_dot_notation_access(self):
        """Test accessing values via dot notation."""
        data = {"mean": [1, 2, 3], "std": [0.1, 0.2, 0.3]}
        result = FeatureResult(data)

        # Test getting values
        assert result.mean == [1, 2, 3]
        assert result.std == [0.1, 0.2, 0.3]

        # Test setting values
        result.new_feature = [4, 5, 6]
        assert result.new_feature == [4, 5, 6]
        assert result["new_feature"] == [4, 5, 6]

    def test_dictionary_access(self):
        """Test accessing values via dictionary access."""
        data = {"mean": [1, 2, 3], "std": [0.1, 0.2, 0.3]}
        result = FeatureResult(data)

        assert result["mean"] == [1, 2, 3]
        assert result["std"] == [0.1, 0.2, 0.3]

    def test_attribute_error(self):
        """Test AttributeError for non-existent attributes."""
        result = FeatureResult({"mean": [1, 2, 3]})

        with pytest.raises(AttributeError):
            _ = result.nonexistent

    def test_delete_attribute(self):
        """Test deleting attributes via dot notation."""
        result = FeatureResult({"mean": [1, 2, 3], "std": [0.1, 0.2, 0.3]})

        del result.std
        assert "std" not in result
        assert "mean" in result

    def test_to_dict(self):
        """Test converting to regular dictionary."""
        data = {"mean": [1, 2, 3], "std": [0.1, 0.2, 0.3]}
        result = FeatureResult(data)
        converted = result.to_dict()

        assert converted == data
        assert type(converted) == dict

    def test_to_numpy(self):
        """Test converting values to numpy arrays."""
        data = {"mean": [1, 2, 3], "std": [0.1, 0.2, 0.3], "label": "test"}
        result = FeatureResult(data)
        numpy_result = result.to_numpy()

        assert isinstance(numpy_result["mean"], np.ndarray)
        assert isinstance(numpy_result["std"], np.ndarray)
        assert numpy_result["label"] == "test"  # Non-convertible values stay as-is


class TestFeatureResultCollection:
    """Tests for FeatureResultCollection class."""

    def test_feature_result_collection_creation(self):
        """Test creating FeatureResultCollection."""
        collection = FeatureResultCollection()
        assert isinstance(collection, dict)
        assert len(collection) == 0

    def test_add_result(self):
        """Test adding results to collection."""
        collection = FeatureResultCollection()
        extractor = MeanExtractor()
        data = {"mean": np.array([1, 2, 3])}

        collection.add_result("mean_extractor", data, extractor, ["input1"])

        assert "mean_extractor" in collection
        assert collection.mean_extractor.data == data
        assert collection.mean_extractor.extractor == extractor
        assert collection.mean_extractor.inputs_used == ["input1"]

    def test_dot_notation_access_collection(self):
        """Test accessing extractor results via dot notation."""
        collection = FeatureResultCollection()
        extractor = MeanExtractor()
        data = {"mean": np.array([1, 2, 3])}

        collection.add_result("mean_extractor", data, extractor)

        # Dot notation access
        assert collection.mean_extractor.data == data
        assert collection.mean_extractor.extractor == extractor

        # Dictionary access
        assert collection["mean_extractor"]["data"] == data
        assert collection["mean_extractor"]["extractor"] == extractor

    def test_get_all_features(self):
        """Test extracting all feature data."""
        collection = FeatureResultCollection()

        collection.add_result("mean_ext", {"mean": [1, 2, 3]})
        collection.add_result("std_ext", {"std": [0.1, 0.2, 0.3]})

        features = collection.get_all_features()
        expected = {"mean": [1, 2, 3], "std": [0.1, 0.2, 0.3]}
        assert features == expected

    def test_get_metadata(self):
        """Test extracting metadata."""
        collection = FeatureResultCollection()
        extractor = MeanExtractor()

        collection.add_result("mean_ext", {"mean": [1, 2, 3]}, extractor, ["input1"])

        metadata = collection.get_metadata()
        assert "mean_ext" in metadata
        assert metadata["mean_ext"]["extractor_type"] == "MeanExtractor"
        assert metadata["mean_ext"]["inputs_used"] == ["input1"]
        assert metadata["mean_ext"]["feature_keys"] == ["mean"]

    def test_get_extractor_names(self):
        """Test getting list of extractor names."""
        collection = FeatureResultCollection()

        collection.add_result("mean_ext", {"mean": [1, 2, 3]})
        collection.add_result("std_ext", {"std": [0.1, 0.2, 0.3]})

        names = collection.get_extractor_names()
        assert set(names) == {"mean_ext", "std_ext"}

    def test_to_dict(self):
        """Test converting collection to regular dictionary."""
        collection = FeatureResultCollection()
        extractor = MeanExtractor()

        collection.add_result("mean_ext", {"mean": [1, 2, 3]}, extractor)

        dict_result = collection.to_dict()
        assert isinstance(dict_result, dict)
        assert "mean_ext" in dict_result
        assert dict_result["mean_ext"]["data"] == {"mean": [1, 2, 3]}
        assert dict_result["mean_ext"]["extractor"] == extractor

    def test_mixed_access_patterns(self):
        """Test mixing dot notation and dictionary access."""
        collection = FeatureResultCollection()

        collection.add_result("mean_ext", {"mean": [1, 2, 3]})
        collection["std_ext"] = FeatureResult(
            {"data": {"std": [0.1, 0.2, 0.3]}, "extractor": None, "inputs_used": []}
        )

        # Mixed access should work
        assert collection.mean_ext.data == {"mean": [1, 2, 3]}
        assert collection["std_ext"]["data"] == {"std": [0.1, 0.2, 0.3]}
        assert collection.std_ext.data == {"std": [0.1, 0.2, 0.3]}
        assert collection["mean_ext"]["data"] == {"mean": [1, 2, 3]}

    def test_iteration(self):
        """Test iterating over collection."""
        collection = FeatureResultCollection()

        collection.add_result("mean_ext", {"mean": [1, 2, 3]})
        collection.add_result("std_ext", {"std": [0.1, 0.2, 0.3]})

        # Test iteration preserves order and functionality
        names = []
        for name, result in collection.items():
            names.append(name)
            assert isinstance(result, FeatureResult)
            assert "data" in result

        assert set(names) == {"mean_ext", "std_ext"}
