"""HSI fixtures."""

import numpy as np
import pytest

from hyppo.core import HSI


@pytest.fixture
def sample_hsi_data():
    """Create a sample HSI data array for testing."""
    # Create synthetic hyperspectral data: 10x10 spatial, 50 spectral bands
    spatial_size = (10, 10)
    spectral_bands = 50

    # Generate synthetic reflectance data
    reflectance = np.random.rand(*spatial_size, spectral_bands) * 0.8

    # Generate synthetic wavelengths (nm)
    wavelengths = np.linspace(400, 1000, spectral_bands)

    return reflectance, wavelengths


@pytest.fixture
def sample_hsi(sample_hsi_data):
    """Create a sample HSI object for testing."""
    reflectance, wavelengths = sample_hsi_data
    return HSI(reflectance=reflectance, wavelengths=wavelengths)


@pytest.fixture
def small_hsi():
    """Create a small HSI object for quick tests."""
    # Create minimal test data: 3x3 spatial, 5 spectral bands
    reflectance = np.array(
        [
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5, 0.6, 0.7],
            ],
            [
                [0.4, 0.5, 0.6, 0.7, 0.8],
                [0.5, 0.6, 0.7, 0.8, 0.9],
                [0.6, 0.7, 0.8, 0.9, 1.0],
            ],
            [
                [0.7, 0.8, 0.9, 1.0, 0.9],
                [0.8, 0.9, 1.0, 0.9, 0.8],
                [0.9, 1.0, 0.9, 0.8, 0.7],
            ],
        ]
    )

    wavelengths = np.array([450, 550, 650, 750, 850])

    return HSI(reflectance=reflectance, wavelengths=wavelengths)
