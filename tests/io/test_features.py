"""Tests for FeatureCollection save functionality."""

from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest

from hyppo import io
from hyppo.core import Feature, FeatureCollection
from hyppo.extractor import NDVIExtractor, SAVIExtractor


class TestSaveFeatureCollection:
    """Test cases for save_feature_collection function."""

    @pytest.fixture
    def sample_feature_collection(self, small_hsi):
        """Create a sample FeatureCollection for testing."""
        # Arrange: Create extractors and extract features
        mean_ext = NDVIExtractor()
        std_ext = SAVIExtractor()

        mean_result = mean_ext.extract(small_hsi)
        std_result = std_ext.extract(small_hsi)

        # Create Feature objects
        mean_feature = Feature(mean_result, mean_ext, {})
        std_feature = Feature(std_result, std_ext, {})

        return FeatureCollection({"n_d_v_i": mean_feature, "s_a_v_i": std_feature})

    def test_save_feature_collection_creates_file(
        self, sample_feature_collection
    ):
        """Test that save_feature_collection creates an HDF5 file."""
        # Arrange: Create temporary file path
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Save the collection
            io.save_feature_collection(sample_feature_collection, tmp_path)

            # Assert: File exists
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_invalid_extension(
        self, sample_feature_collection
    ):
        """Test save_feature_collection raises error for invalid extension."""
        # Arrange: Create path with wrong extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act & Assert: Verify ValueError is raised
            with pytest.raises(ValueError, match=r".*\.h5.*"):
                io.save_feature_collection(sample_feature_collection, tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_empty_collection(self):
        """Test save_feature_collection raises error for empty collection."""
        # Arrange: Create empty collection
        empty_collection = FeatureCollection({})

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act & Assert: Verify ValueError is raised
            with pytest.raises(ValueError, match="empty"):
                io.save_feature_collection(empty_collection, tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_saves_features(
        self, sample_feature_collection
    ):
        """Test that feature arrays are saved correctly."""
        # Arrange: Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Save collection
            io.save_feature_collection(sample_feature_collection, tmp_path)

            # Assert: Verify features group and feature datasets exist
            with h5py.File(tmp_path, "r") as f:
                assert "features" in f
                assert "n_d_v_i" in f["features"]
                assert "s_a_v_i" in f["features"]

                # Verify ndvi feature data
                assert "features" in f["features/n_d_v_i"]
                ndvi_data = f["features/n_d_v_i/features"][:]
                expected_ndvi = sample_feature_collection["n_d_v_i"].data[
                    "features"
                ]
                np.testing.assert_array_equal(ndvi_data, expected_ndvi)

                # Verify savi feature data
                assert "features" in f["features/s_a_v_i"]
                savi_data = f["features/s_a_v_i/features"][:]
                expected_savi = sample_feature_collection["s_a_v_i"].data[
                    "features"
                ]
                np.testing.assert_array_equal(savi_data, expected_savi)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_saves_metadata(
        self, sample_feature_collection
    ):
        """Test that metadata is saved correctly."""
        # Arrange: Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Save collection
            io.save_feature_collection(sample_feature_collection, tmp_path)

            # Assert: Verify metadata group exists
            with h5py.File(tmp_path, "r") as f:
                assert "metadata" in f
                assert "n_d_v_i" in f["metadata"]
                assert "s_a_v_i" in f["metadata"]

                # Verify ndvi metadata
                ndvi_meta = f["metadata/n_d_v_i"]
                assert "extractor_type" in ndvi_meta.attrs
                assert ndvi_meta.attrs["extractor_type"] == "NDVIExtractor"

                # Verify savi metadata
                savi_meta = f["metadata/s_a_v_i"]
                assert "extractor_type" in savi_meta.attrs
                assert savi_meta.attrs["extractor_type"] == "SAVIExtractor"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_with_extra_data(self, small_hsi):
        """Test saving features with extra data beyond 'features' key."""
        # Arrange: Create feature with extra data
        mean_ext = NDVIExtractor()
        result = mean_ext.extract(small_hsi)
        result["extra_key"] = np.array([1, 2, 3])
        result["another_key"] = np.array([[4, 5], [6, 7]])

        feature = Feature(result, mean_ext, {})
        collection = FeatureCollection({"n_d_v_i": feature})

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Save collection
            io.save_feature_collection(collection, tmp_path)

            # Assert: Verify extra data is saved
            with h5py.File(tmp_path, "r") as f:
                assert "extra_key" in f["features/n_d_v_i"]
                assert "another_key" in f["features/n_d_v_i"]

                np.testing.assert_array_equal(
                    f["features/n_d_v_i/extra_key"][:], [1, 2, 3]
                )
                np.testing.assert_array_equal(
                    f["features/n_d_v_i/another_key"][:], [[4, 5], [6, 7]]
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_accepts_string_path(
        self, sample_feature_collection
    ):
        """Test that save_feature_collection accepts string paths."""
        # Arrange: Create string path
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Save with string path
            io.save_feature_collection(sample_feature_collection, tmp_path)

            # Assert: File created
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_accepts_path_object(
        self, sample_feature_collection
    ):
        """Test that save_feature_collection accepts Path objects."""
        # Arrange: Create Path object
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Act: Save with Path object
            io.save_feature_collection(sample_feature_collection, tmp_path)

            # Assert: File created
            assert tmp_path.exists()
        finally:
            tmp_path.unlink(missing_ok=True)


class TestFeatureCollectionSaveMethod:
    """Test cases for FeatureCollection.save() method."""

    @pytest.fixture
    def sample_feature_collection(self, small_hsi):
        """Create a sample FeatureCollection for testing."""
        # Arrange: Create extractors and extract features
        mean_ext = NDVIExtractor()
        mean_result = mean_ext.extract(small_hsi)
        mean_feature = Feature(mean_result, mean_ext, {})

        return FeatureCollection({"n_d_v_i": mean_feature})

    def test_save_method_creates_file(self, sample_feature_collection):
        """Test that FeatureCollection.save() creates an HDF5 file."""
        # Arrange: Create temporary file path
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Call save method
            sample_feature_collection.save(tmp_path)

            # Assert: File exists
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_method_calls_save_feature_collection(
        self, sample_feature_collection
    ):
        """Test that save() method delegates to save_feature_collection."""
        # Arrange: Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Call save method
            sample_feature_collection.save(tmp_path)

            # Assert: Verify file structure matches expected behavior
            with h5py.File(tmp_path, "r") as f:
                assert "features" in f
                assert "metadata" in f
                assert "n_d_v_i" in f["features"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_method_invalid_extension(self, sample_feature_collection):
        """Test that save() raises ValueError for invalid extension."""
        # Arrange: Create path with wrong extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act & Assert: Verify ValueError is raised
            with pytest.raises(ValueError, match=r".*\.h5.*"):
                sample_feature_collection.save(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_without_dict_data(self, small_hsi):
        """Test saving feature whose data is not a dict."""
        # Arrange: Create Feature with non-dict data
        from hyppo.utils.bunch import Bunch

        feature = Bunch("Feature", {
            "result": None,
            "data": "not_a_dict",
            "extractor": None,
            "inputs_used": [],
        })
        collection = FeatureCollection({"test": feature})

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Save collection with non-dict data feature
            io.save_feature_collection(collection, tmp_path)

            # Assert: File created, feature group exists but is empty
            with h5py.File(tmp_path, "r") as f:
                assert "features" in f
                assert "test" in f["features"]
                assert len(f["features"]["test"]) == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)
