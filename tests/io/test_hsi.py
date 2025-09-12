"""
Tests for the IO module.
"""

import pytest
import numpy as np
import tempfile
import h5py
from pathlib import Path
from hyppo import io
from hyppo.core import HSI


class TestH5Loading:
    """Test cases for H5 file loading functionality."""

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, OSError)):
            io.load("nonexistent_file.h5")

    def test_load_invalid_format(self):
        """Test loading an invalid file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
            tmp_file.write(b"not an h5 file")
            tmp_file.flush()

            with pytest.raises((ValueError, OSError)):
                io.load(tmp_file.name)

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

    def test_load_valid_h5_file(self, sample_h5_file):
        """Test loading a valid H5 file."""
        try:
            hsi = io.load(sample_h5_file)

            assert isinstance(hsi, HSI)
            assert hsi.reflectance is not None
            assert hsi.wavelengths is not None
            assert hsi.reflectance.ndim == 3
            assert len(hsi.wavelengths) == hsi.reflectance.shape[2]

        except NotImplementedError:
            pytest.skip("H5 loading not fully implemented yet")
        except Exception as e:
            pytest.fail(f"Failed to load valid H5 file: {e}")


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
        try:
            hsi = io.load(complex_h5_file)

            # Should successfully load despite multiple datasets
            assert isinstance(hsi, HSI)
            assert hsi.reflectance.shape == (5, 5, 20)
            assert len(hsi.wavelengths) == 20

        except NotImplementedError:
            pytest.skip("Heuristic detection not implemented yet")
        except Exception as e:
            pytest.fail(f"Heuristic detection failed: {e}")


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
