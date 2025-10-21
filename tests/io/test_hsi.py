"""
Tests for the IO module.
"""

import h5py
from hyppo import io
from hyppo.core import HSI
import numpy as np
from pathlib import Path
import pytest
import tempfile


class TestH5Loading:
    """Test cases for H5 file loading functionality."""

    def test_load_h5_hsi_nonexistent_file(self):
        """Test loading a non-existent H5 file raises appropriate error."""
        with pytest.raises((FileNotFoundError, OSError)):
            io.load_h5_hsi("nonexistent_file.h5")

    def test_load_h5_hsi_invalid_format(self):
        """Test loading an invalid file format with load_h5_hsi."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
            tmp_file.write(b"not an h5 file")
            tmp_file.flush()

            with pytest.raises(ValueError, match="Unknown Hyper Spectral Image format"):
                io.load_h5_hsi(tmp_file.name)

    @pytest.fixture
    def sample_h5_file(self):
        """Create a sample H5 file for testing."""
        # Create synthetic HSI data
        spatial_shape = (10, 10)
        spectral_bands = 50
        reflectance = np.random.rand(*spatial_shape, spectral_bands) * 0.8
        wavelengths = np.linspace(400, 1000, spectral_bands)

        # Create temporary H5 file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                # Create datasets that match expected heuristics
                f.create_dataset("reflectance_data", data=reflectance)
                f.create_dataset("wavelength_data", data=wavelengths)

                # Add some metadata
                f.attrs["description"] = "Test hyperspectral data"

            yield tmp_path

        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_h5_hsi_valid_file(self, sample_h5_file):
        """Test loading a valid H5 file with load_h5_hsi."""
        hsi = io.load_h5_hsi(sample_h5_file)

        assert isinstance(hsi, HSI)
        assert hsi.reflectance is not None
        assert hsi.wavelengths is not None
        assert hsi.reflectance.ndim == 3
        assert len(hsi.wavelengths) == hsi.reflectance.shape[2]
        assert hsi.reflectance.dtype == np.float32
        assert hsi.wavelengths.dtype == np.float32
        assert hsi.mask is not None
        assert hsi.mask.dtype == bool


class TestHeuristicDetection:
    """Test cases for dataset detection heuristics."""

    @pytest.fixture
    def complex_h5_file(self):
        """Create a complex H5 file with multiple datasets."""
        spatial_shape = (5, 5)
        spectral_bands = 20
        reflectance = np.random.rand(*spatial_shape, spectral_bands) * 0.8
        radiance = reflectance * 1000  # Simulated radiance
        wavelengths = np.linspace(450, 900, spectral_bands)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                # Create multiple datasets with different names
                f.create_dataset("sensor/reflectance_cube", data=reflectance)
                f.create_dataset("sensor/radiance_cube", data=radiance)
                f.create_dataset("metadata/wavelengths", data=wavelengths)
                f.create_dataset("other_data", data=np.random.rand(10, 10))

                # Add scale factors and null values
                refl_ds = f["sensor/reflectance_cube"]
                refl_ds.attrs["scale_factor"] = 0.0001
                refl_ds.attrs["_FillValue"] = -9999

            yield tmp_path

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_heuristic_dataset_detection(self, complex_h5_file):
        """Test that heuristics correctly identify datasets."""
        hsi = io.load_h5_hsi(complex_h5_file)

        # Should successfully load despite multiple datasets
        assert isinstance(hsi, HSI)
        assert hsi.reflectance.shape == (5, 5, 20)
        assert len(hsi.wavelengths) == 20


class TestDataValidation:
    """Test cases for data validation during loading."""

    def test_reflectance_range_validation(self):
        """Test validation of reflectance value ranges."""
        # This test assumes reflectance should be in [0, 1] range
        reflectance = np.random.rand(5, 5, 10)
        wavelengths = np.linspace(400, 800, 10)

        # Valid data should work
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        assert np.all(hsi.reflectance >= 0)
        assert np.all(hsi.reflectance <= 1)

    def test_wavelength_validation(self):
        """Test validation of wavelength values."""
        reflectance = np.random.rand(5, 5, 10) * 0.8

        # Test with reasonable wavelengths
        wavelengths = np.linspace(400, 1000, 10)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        assert np.all(hsi.wavelengths > 0)
        assert np.all(hsi.wavelengths < 10000)  # Reasonable upper bound


class TestLoadH5Specific:
    """Test cases specifically for the load_h5_hsi function."""

    def test_load_h5_hsi_with_explicit_paths(self):
        """Test load_h5_hsi with explicit dataset paths."""
        spatial_shape = (5, 5)
        spectral_bands = 10
        reflectance = np.random.rand(*spatial_shape, spectral_bands) * 0.8
        wavelengths = np.linspace(400, 800, spectral_bands)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("custom/reflectance", data=reflectance)
                f.create_dataset("custom/wavelengths", data=wavelengths)

            # Load with explicit paths
            hsi = io.load_h5_hsi(
                tmp_path,
                reflectance_path="custom/reflectance",
                wavelength_path="custom/wavelengths",
            )

            assert isinstance(hsi, HSI)
            assert hsi.reflectance.shape == reflectance.shape
            assert len(hsi.wavelengths) == len(wavelengths)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_h5_hsi_with_scale_factor(self):
        """Test load_h5_hsi with scale factor handling."""
        spatial_shape = (3, 3)
        spectral_bands = 5
        # Create scaled integer data
        reflectance_raw = (
            np.random.rand(*spatial_shape, spectral_bands) * 10000
        ).astype(np.int16)
        wavelengths = np.linspace(500, 700, spectral_bands)
        scale_factor = 10000.0

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                ds = f.create_dataset("reflectance_scaled", data=reflectance_raw)
                ds.attrs["Scale_Factor"] = [scale_factor]
                f.create_dataset("wavelength", data=wavelengths)

            hsi = io.load_h5_hsi(tmp_path)

            # Check that scale factor was applied
            assert isinstance(hsi, HSI)
            assert hsi.reflectance.dtype == np.float32
            expected_max = reflectance_raw.max() / scale_factor
            assert np.isclose(hsi.reflectance.max(), expected_max, rtol=1e-5)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_h5_hsi_with_null_values(self):
        """Test load_h5_hsi with null value handling."""
        spatial_shape = (4, 4)
        spectral_bands = 5
        reflectance = np.random.rand(*spatial_shape, spectral_bands) * 0.8
        wavelengths = np.linspace(400, 800, spectral_bands)

        # Set some pixels to null value
        null_value = -9999.0
        reflectance[0, 0, :] = null_value  # First pixel all bands null
        reflectance[1, 1, 2] = null_value  # One band null in second pixel

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                ds = f.create_dataset("reflectance_with_nulls", data=reflectance)
                ds.attrs["Data_Ignore_Value"] = [null_value]
                f.create_dataset("wavelength", data=wavelengths)

            hsi = io.load_h5_hsi(tmp_path)

            # Check mask creation - a pixel is valid only if ALL bands are valid
            assert isinstance(hsi, HSI)
            assert (
                hsi.mask[0, 0] == False
            )  # First pixel should be masked (all bands null)
            assert (
                hsi.mask[1, 1] == False
            )  # Second pixel should be masked (one band null)
            assert (
                np.sum(hsi.mask) < spatial_shape[0] * spatial_shape[1]
            )  # Some pixels masked

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_h5_hsi_missing_datasets(self):
        """Test load_h5_hsi with missing required datasets."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                # Create file with no datasets
                f.attrs["description"] = "Empty H5 file"

            with pytest.raises(ValueError, match="Could not find reflectance dataset"):
                io.load_h5_hsi(tmp_path)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_h5_hsi_invalid_dataset_paths(self):
        """Test load_h5_hsi with invalid explicit dataset paths."""
        spatial_shape = (3, 3)
        spectral_bands = 5
        reflectance = np.random.rand(*spatial_shape, spectral_bands) * 0.8
        wavelengths = np.linspace(400, 800, spectral_bands)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("reflectance_data", data=reflectance)
                f.create_dataset("wavelength_data", data=wavelengths)

            # Test with invalid reflectance path
            with pytest.raises(
                ValueError, match="Provided reflectance path .* is invalid"
            ):
                io.load_h5_hsi(tmp_path, reflectance_path="nonexistent/path")

            # Test with invalid wavelength path
            with pytest.raises(
                ValueError, match="Provided wavelength path .* is invalid"
            ):
                io.load_h5_hsi(tmp_path, wavelength_path="nonexistent/path")

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_h5_hsi_missing_wavelength_dataset(self):
        """Test load_h5_hsi with missing wavelength dataset."""
        spatial_shape = (3, 3)
        spectral_bands = 5
        reflectance = np.random.rand(*spatial_shape, spectral_bands) * 0.8

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("reflectance_data", data=reflectance)

            with pytest.raises(ValueError, match="Could not find wavelength dataset"):
                io.load_h5_hsi(tmp_path)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_h5_hsi_with_bytes_metadata(self):
        """Test load_h5_hsi processes bytes metadata without errors."""
        spatial_shape = (3, 3)
        spectral_bands = 5
        reflectance = np.random.rand(*spatial_shape, spectral_bands) * 0.8
        wavelengths = np.linspace(400, 800, spectral_bands)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                ds = f.create_dataset("reflectance_data", data=reflectance)
                ds.attrs["description"] = b"Test bytes metadata"
                ds.attrs["sensor"] = "HSI Sensor"
                f.create_dataset("wavelength", data=wavelengths)

            hsi = io.load_h5_hsi(tmp_path)

            assert isinstance(hsi, HSI)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_h5_hsi_with_string_array_metadata(self):
        """Test load_h5_hsi with string array metadata attributes."""
        spatial_shape = (3, 3)
        spectral_bands = 5
        reflectance = np.random.rand(*spatial_shape, spectral_bands) * 0.8
        wavelengths = np.linspace(400, 800, spectral_bands)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                ds = f.create_dataset("reflectance_data", data=reflectance)
                ds.attrs["band_names"] = np.array(
                    [b"band1", b"band2", b"band3"], dtype="S10"
                )
                f.create_dataset("wavelength", data=wavelengths)

            hsi = io.load_h5_hsi(tmp_path)

            assert isinstance(hsi, HSI)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_h5_hsi_with_invalid_metadata_encoding(self):
        """Test load_h5_hsi with invalid UTF-8 metadata."""
        spatial_shape = (3, 3)
        spectral_bands = 5
        reflectance = np.random.rand(*spatial_shape, spectral_bands) * 0.8
        wavelengths = np.linspace(400, 800, spectral_bands)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with h5py.File(tmp_path, "w") as f:
                ds = f.create_dataset("reflectance_data", data=reflectance)
                ds.attrs.create("invalid_utf8", b"\xff\xfe", dtype=h5py.string_dtype())
                f.create_dataset("wavelength", data=wavelengths)

            hsi = io.load_h5_hsi(tmp_path)

            assert isinstance(hsi, HSI)

        finally:
            Path(tmp_path).unlink(missing_ok=True)
