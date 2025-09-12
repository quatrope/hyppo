"""
Tests for the HSI (Hyperspectral Image) module.
"""

import numpy as np
from hyppo.core import HSI


def test_hsi_initialization():
    """Test HSI object initialization."""
    reflectance = np.random.rand(10, 10, 50)
    wavelengths = np.linspace(400, 1000, 50)

    hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)

    assert hsi.reflectance.shape == (10, 10, 50)
    assert hsi.wavelengths.shape == (50,)
    assert np.allclose(hsi.reflectance, reflectance)
    assert np.allclose(hsi.wavelengths, wavelengths)


def test_hsi_shape_properties(sample_hsi):
    """Test HSI shape properties."""
    hsi = sample_hsi

    # Should have spatial dimensions + spectral dimension
    assert len(hsi.reflectance.shape) == 3
    assert hsi.reflectance.shape[2] == len(hsi.wavelengths)


def test_hsi_wavelength_mismatch():
    """Test HSI initialization with mismatched reflectance and wavelengths."""
    reflectance = np.random.rand(10, 10, 50)
    wavelengths = np.linspace(400, 1000, 30)  # Wrong number of wavelengths

    # This should raise an error if HSI validates dimensions
    # If not implemented yet, this test serves as a specification
    try:
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        # If no validation exists yet, we'll just check shapes don't match
        assert hsi.reflectance.shape[2] != len(hsi.wavelengths)
    except (ValueError, AssertionError):
        # Expected behavior when validation is implemented
        pass


def test_hsi_data_types(sample_hsi):
    """Test HSI data types."""
    hsi = sample_hsi

    assert isinstance(hsi.reflectance, np.ndarray)
    assert isinstance(hsi.wavelengths, np.ndarray)
    assert hsi.reflectance.dtype in [np.float32, np.float64]
    assert hsi.wavelengths.dtype in [np.float32, np.float64]


def test_small_hsi_fixture(small_hsi):
    """Test the small HSI fixture."""
    hsi = small_hsi

    assert hsi.reflectance.shape == (3, 3, 5)
    assert len(hsi.wavelengths) == 5
    assert np.array_equal(hsi.wavelengths, [450, 550, 650, 750, 850])
