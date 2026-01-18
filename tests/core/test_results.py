"""Tests for Feature and FeatureCollection."""

import numpy as np
import pandas as pd

from hyppo.core import Feature, FeatureCollection
from hyppo.extractor.ndvi import NDVIExtractor


class TestFeature:
    """Tests for Feature class."""

    def test_feature_result_creation(self):
        """Test creating Feature from dictionary."""
        data = {"n_d_v_i": [1, 2, 3], "s_a_v_i": [0.1, 0.2, 0.3]}
        result = Feature(data, extractor=None, inputs_used=[])

        assert result.to_dict() == {
            "result": None,
            "data": data,
            "extractor": None,
            "inputs_used": [],
        }

    def test_describe_with_features(self):
        """Test describe method with features array."""
        # Arrange: Create result with features array
        data = {
            "features": np.array([[1, 2, 3], [4, 5, 6]]),
            "extra1": "value1",
            "extra2": "value2",
        }
        result = Feature(data, extractor=None, inputs_used=[])

        # Act: Get description
        desc = result.describe()

        # Assert: Verify description
        assert desc["dimensions"] == (2, 3)
        assert desc["extra_data"] == "extra1, extra2"

    def test_describe_without_features(self):
        """Test describe method without features array."""
        # Arrange: Create result without features
        data = {"key1": "value1", "key2": "value2"}
        result = Feature(data, extractor=None, inputs_used=[])

        # Act: Get description
        desc = result.describe()

        # Assert: Verify description
        assert desc["dimensions"] is None
        assert desc["extra_data"] == "key1, key2"

    def test_describe_empty_data(self):
        """Test describe method with empty data."""
        # Arrange: Create result with empty data
        result = Feature({}, extractor=None, inputs_used=[])

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

        data = {"features": ShapeObject()}
        result = Feature(data, extractor=None, inputs_used=[])

        # Act: Get description
        desc = result.describe()

        # Assert: Verify description uses shape attribute
        assert desc["dimensions"] == (5, 10)
        assert desc["extra_data"] == ""


class TestFeatureCollection:
    """Tests for FeatureCollection class."""

    def test_feature_result_collection_creation(self):
        """Test creating FeatureCollection."""
        collection = FeatureCollection.from_features({})
        assert len(collection) == 0

    def test_add_result(self):
        """Test adding results to collection."""
        extractor = NDVIExtractor()
        data = {"n_d_v_i": np.array([1, 2, 3])}

        collection = FeatureCollection.from_features(
            {"mean_extractor": Feature(data, extractor, ["input1"])}
        )

        assert "mean_extractor" in collection
        assert collection.mean_extractor.data == data
        assert collection.mean_extractor.extractor == extractor
        assert collection.mean_extractor.inputs_used == ["input1"]

    def test_get_all_features(self):
        """Test extracting all feature data."""
        collection = FeatureCollection.from_features(
            {
                "mean_ext": Feature({"n_d_v_i": [1, 2, 3]}, None, []),
                "std_ext": Feature({"s_a_v_i": [0.1, 0.2, 0.3]}, None, []),
            }
        )

        features = collection.get_all_features()
        expected = {"n_d_v_i": [1, 2, 3], "s_a_v_i": [0.1, 0.2, 0.3]}
        assert features == expected

    def test_get_metadata(self):
        """Test extracting metadata."""
        extractor = NDVIExtractor()

        collection = FeatureCollection.from_features(
            {"mean_ext": Feature({"n_d_v_i": [1, 2, 3]}, extractor, ["input1"])}
        )

        metadata = collection.get_metadata()
        assert "mean_ext" in metadata
        assert metadata["mean_ext"]["extractor_type"] == "NDVIExtractor"
        assert metadata["mean_ext"]["inputs_used"] == ["input1"]
        assert metadata["mean_ext"]["feature_keys"] == ["n_d_v_i"]

    def test_get_extractor_names(self):
        """Test getting list of extractor names."""
        collection = FeatureCollection.from_features(
            {
                "mean_ext": Feature({"n_d_v_i": [1, 2, 3]}, None, []),
                "std_ext": Feature({"s_a_v_i": [0.1, 0.2, 0.3]}, None, []),
            }
        )

        names = collection.get_extractor_names()
        assert set(names) == {"mean_ext", "std_ext"}

    def test_to_dict(self):
        """Test converting collection to dictionary."""
        # Arrange: Create collection with results
        collection = FeatureCollection.from_features(
            {
                "mean_ext": Feature({"n_d_v_i": [1, 2, 3]}, None, []),
                "std_ext": Feature({"s_a_v_i": [0.1, 0.2, 0.3]}, None, []),
            }
        )

        # Act: Convert to dict
        result_dict = collection.to_dict()

        # Assert: Verify structure
        assert "mean_ext" in result_dict
        assert "std_ext" in result_dict
        assert isinstance(result_dict, dict)

    def test_describe_basic(self):
        """Test describe method with multiple feature results."""
        # Arrange: Create collection with results
        collection = FeatureCollection.from_features(
            {
                "mean_ext": Feature(
                    {
                        "features": np.array([[1, 2, 3], [4, 5, 6]]),
                        "extra": "value",
                    },
                    None,
                    [],
                ),
                "std_ext": Feature(
                    {"features": np.array([1, 2, 3, 4])}, None, []
                ),
            }
        )

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
        collection = FeatureCollection({})

        # Act: Get description
        df = collection.describe()

        # Assert: Verify empty DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["feature_name", "dimensions", "extra_data"]
