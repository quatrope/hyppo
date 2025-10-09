import pytest
import numpy as np
import pandas as pd
from hyppo.core import FeatureResult, FeatureResultCollection
from hyppo.extractor.mean import MeanExtractor


class TestFeatureResult:
    """Tests for FeatureResult class."""

    def test_feature_result_creation(self):
        """Test creating FeatureResult from dictionary."""
        data = {"mean": [1, 2, 3], "std": [0.1, 0.2, 0.3]}
        result = FeatureResult(data)

        assert result == data

    def test_to_numpy(self):
        """Test converting values to numpy arrays."""
        # Arrange: Create result with mixed types including non-convertible
        class BadArray:
            """Object that raises on array conversion."""
            def __array__(self):
                raise ValueError("Cannot convert")

        bad_obj = BadArray()
        data = {
            "mean": [1, 2, 3],
            "std": [0.1, 0.2, 0.3],
            "label": "test",
            "obj": bad_obj
        }
        result = FeatureResult(data)

        # Act: Convert to numpy
        numpy_result = result.to_numpy()

        # Assert: Verify conversions
        assert isinstance(numpy_result["mean"], np.ndarray)
        assert isinstance(numpy_result["std"], np.ndarray)
        assert numpy_result["label"] == "test"
        assert numpy_result["obj"] is bad_obj

    def test_describe_with_features(self):
        """Test describe method with features array."""
        # Arrange: Create result with features array
        data = {
            "data": {
                "features": np.array([[1, 2, 3], [4, 5, 6]]),
                "extra1": "value1",
                "extra2": "value2"
            }
        }
        result = FeatureResult(data)

        # Act: Get description
        desc = result.describe()

        # Assert: Verify description
        assert desc["dimensions"] == (2, 3)
        assert desc["extra_data"] == "extra1, extra2"

    def test_describe_without_features(self):
        """Test describe method without features array."""
        # Arrange: Create result without features
        data = {"data": {"key1": "value1", "key2": "value2"}}
        result = FeatureResult(data)

        # Act: Get description
        desc = result.describe()

        # Assert: Verify description
        assert desc["dimensions"] is None
        assert desc["extra_data"] == "key1, key2"

    def test_describe_empty_data(self):
        """Test describe method with empty data."""
        # Arrange: Create result with empty data
        result = FeatureResult({})

        # Act: Get description
        desc = result.describe()

        # Assert: Verify description
        assert desc["dimensions"] is None
        assert desc["extra_data"] == ""

    def test_describe_with_shape_attribute(self):
        """Test describe method with non-ndarray object that has shape."""
        # Arrange: Create mock object with shape attribute
        class ShapeObject:
            """Mock object with shape attribute."""
            shape = (5, 10)

        data = {"data": {"features": ShapeObject()}}
        result = FeatureResult(data)

        # Act: Get description
        desc = result.describe()

        # Assert: Verify description uses shape attribute
        assert desc["dimensions"] == (5, 10)
        assert desc["extra_data"] == ""


class TestFeatureResultCollection:
    """Tests for FeatureResultCollection class."""

    def test_feature_result_collection_creation(self):
        """Test creating FeatureResultCollection."""
        collection = FeatureResultCollection({})
        assert len(collection) == 0

    def test_add_result(self):
        """Test adding results to collection."""
        collection = FeatureResultCollection({})
        extractor = MeanExtractor()
        data = {"mean": np.array([1, 2, 3])}

        collection.add_result("mean_extractor", data, extractor, ["input1"])

        assert "mean_extractor" in collection
        assert collection.mean_extractor.data == data
        assert collection.mean_extractor.extractor == extractor
        assert collection.mean_extractor.inputs_used == ["input1"]

    def test_get_all_features(self):
        """Test extracting all feature data."""
        collection = FeatureResultCollection({})

        collection.add_result("mean_ext", {"mean": [1, 2, 3]})
        collection.add_result("std_ext", {"std": [0.1, 0.2, 0.3]})

        features = collection.get_all_features()
        expected = {"mean": [1, 2, 3], "std": [0.1, 0.2, 0.3]}
        assert features == expected

    def test_get_all_features_with_non_dict_data(self):
        """Test get_all_features with non-dict result data."""
        # Arrange: Create collection with mixed result types
        collection = FeatureResultCollection({})
        collection.add_result("normal", {"feature1": [1, 2, 3]})

        # Add result with non-dict data
        non_dict_result = FeatureResult({"data": [1, 2, 3]})
        collection["non_dict"] = non_dict_result

        # Act: Get all features
        features = collection.get_all_features()

        # Assert: Verify non-dict handled correctly
        assert "feature1" in features
        assert "non_dict" in features
        assert features["non_dict"] == [1, 2, 3]

    def test_get_metadata(self):
        """Test extracting metadata."""
        collection = FeatureResultCollection({})
        extractor = MeanExtractor()

        collection.add_result("mean_ext", {"mean": [1, 2, 3]}, extractor, ["input1"])

        metadata = collection.get_metadata()
        assert "mean_ext" in metadata
        assert metadata["mean_ext"]["extractor_type"] == "MeanExtractor"
        assert metadata["mean_ext"]["inputs_used"] == ["input1"]
        assert metadata["mean_ext"]["feature_keys"] == ["mean"]

    def test_get_extractor_names(self):
        """Test getting list of extractor names."""
        collection = FeatureResultCollection({})

        collection.add_result("mean_ext", {"mean": [1, 2, 3]})
        collection.add_result("std_ext", {"std": [0.1, 0.2, 0.3]})

        names = collection.get_extractor_names()
        assert set(names) == {"mean_ext", "std_ext"}

    def test_to_dict(self):
        """Test converting collection to dictionary."""
        # Arrange: Create collection with results
        collection = FeatureResultCollection({})
        collection.add_result("mean_ext", {"mean": [1, 2, 3]})
        collection.add_result("std_ext", {"std": [0.1, 0.2, 0.3]})

        # Act: Convert to dict
        result_dict = collection.to_dict()

        # Assert: Verify structure
        assert "mean_ext" in result_dict
        assert "std_ext" in result_dict
        assert isinstance(result_dict, dict)

    def test_describe_basic(self):
        """Test describe method with multiple feature results."""
        # Arrange: Create collection with results
        collection = FeatureResultCollection({})
        collection.add_result("mean_ext", {
            "features": np.array([[1, 2, 3], [4, 5, 6]]),
            "extra": "value"
        })
        collection.add_result("std_ext", {
            "features": np.array([1, 2, 3, 4])
        })

        # Act: Get description
        df = collection.describe()

        # Assert: Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["feature_name", "dimensions", "extra_data"]

        # Assert: Verify content
        mean_row = df[df["feature_name"] == "mean_ext"].iloc[0]
        assert mean_row["dimensions"] == (2, 3)
        assert mean_row["extra_data"] == "extra"

        std_row = df[df["feature_name"] == "std_ext"].iloc[0]
        assert std_row["dimensions"] == (4,)
        assert std_row["extra_data"] == ""

    def test_describe_empty_collection(self):
        """Test describe method with empty collection."""
        # Arrange: Create empty collection
        collection = FeatureResultCollection({})

        # Act: Get description
        df = collection.describe()

        # Assert: Verify empty DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["feature_name", "dimensions", "extra_data"]