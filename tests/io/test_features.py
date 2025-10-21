"""Tests for FeatureCollection save functionality."""

import pytest
import h5py
import numpy as np
import tempfile
from pathlib import Path

from hyppo import io
from hyppo.core import FeatureCollection, Feature
from hyppo.extractor import MeanExtractor, StdExtractor


class TestSaveFeatureCollection:
    """Test cases for save_feature_collection function."""

    @pytest.fixture
    def sample_feature_collection(self, small_hsi):
        """Create a sample FeatureCollection for testing."""
        # Arrange: Create extractors and extract features
        mean_ext = MeanExtractor()
        std_ext = StdExtractor()

        mean_result = mean_ext.extract(small_hsi)
        std_result = std_ext.extract(small_hsi)

        # Create Feature objects
        mean_feature = Feature(mean_result, mean_ext, {})
        std_feature = Feature(std_result, std_ext, {})

        return FeatureCollection({
            "mean": mean_feature,
            "std": std_feature
        })

    def test_save_feature_collection_creates_file(self, sample_feature_collection):
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

    def test_save_feature_collection_invalid_extension(self, sample_feature_collection):
        """Test that save_feature_collection raises ValueError for invalid extension."""
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
        """Test that save_feature_collection raises ValueError for empty collection."""
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

    def test_save_feature_collection_saves_features(self, sample_feature_collection):
        """Test that feature arrays are saved correctly."""
        # Arrange: Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Save collection
            io.save_feature_collection(sample_feature_collection, tmp_path)

            # Assert: Verify features group exists and contains feature datasets
            with h5py.File(tmp_path, "r") as f:
                assert "features" in f
                assert "mean" in f["features"]
                assert "std" in f["features"]

                # Verify mean feature data
                assert "features" in f["features/mean"]
                mean_data = f["features/mean/features"][:]
                expected_mean = sample_feature_collection["mean"].data["features"]
                np.testing.assert_array_equal(mean_data, expected_mean)

                # Verify std feature data
                assert "features" in f["features/std"]
                std_data = f["features/std/features"][:]
                expected_std = sample_feature_collection["std"].data["features"]
                np.testing.assert_array_equal(std_data, expected_std)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_saves_metadata(self, sample_feature_collection):
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
                assert "mean" in f["metadata"]
                assert "std" in f["metadata"]

                # Verify mean metadata
                mean_meta = f["metadata/mean"]
                assert "extractor_type" in mean_meta.attrs
                assert mean_meta.attrs["extractor_type"] == "MeanExtractor"

                # Verify std metadata
                std_meta = f["metadata/std"]
                assert "extractor_type" in std_meta.attrs
                assert std_meta.attrs["extractor_type"] == "StdExtractor"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_with_extra_data(self, small_hsi):
        """Test saving features with extra data beyond 'features' key."""
        # Arrange: Create feature with extra data
        mean_ext = MeanExtractor()
        result = mean_ext.extract(small_hsi)
        result["extra_key"] = np.array([1, 2, 3])
        result["another_key"] = np.array([[4, 5], [6, 7]])

        feature = Feature(result, mean_ext, {})
        collection = FeatureCollection({"mean": feature})

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Save collection
            io.save_feature_collection(collection, tmp_path)

            # Assert: Verify extra data is saved
            with h5py.File(tmp_path, "r") as f:
                assert "extra_key" in f["features/mean"]
                assert "another_key" in f["features/mean"]

                np.testing.assert_array_equal(f["features/mean/extra_key"][:], [1, 2, 3])
                np.testing.assert_array_equal(f["features/mean/another_key"][:], [[4, 5], [6, 7]])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_feature_collection_accepts_string_path(self, sample_feature_collection):
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

    def test_save_feature_collection_accepts_path_object(self, sample_feature_collection):
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
        mean_ext = MeanExtractor()
        mean_result = mean_ext.extract(small_hsi)
        mean_feature = Feature(mean_result, mean_ext, {})

        return FeatureCollection({"mean": mean_feature})

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

    def test_save_method_calls_save_feature_collection(self, sample_feature_collection):
        """Test that save() method delegates to save_feature_collection."""
        # Arrange: Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Act: Call save method
            sample_feature_collection.save(tmp_path)

            # Assert: Verify file structure matches save_feature_collection behavior
            with h5py.File(tmp_path, "r") as f:
                assert "features" in f
                assert "metadata" in f
                assert "mean" in f["features"]
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
