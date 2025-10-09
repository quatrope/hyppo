"""Tests for GaborExtractor."""

import pytest
import numpy as np
from hyppo.extractor.gabor import GaborExtractor


class TestGaborExtractor:
    """Test cases for GaborExtractor."""

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = GaborExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result

        # Assert: Verify feature shape matches (H, W, n_features)
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom frequencies and orientations."""
        # Arrange: Create extractor with custom parameters
        frequencies = [0.1, 0.2]
        thetas = [0, np.pi / 2]
        sigma = 2.0
        extractor = GaborExtractor(
            frequencies=frequencies, thetas=thetas, sigma=sigma
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width

    def test_aggregate_bands_true(self, small_hsi):
        """Test feature aggregation across bands."""
        # Arrange: Create extractor with band aggregation
        extractor = GaborExtractor(aggregate_bands=True)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify aggregated features shape
        features = result["features"]
        n_frequencies = 3  # default
        n_orientations = 4  # default
        expected_features = n_frequencies * n_orientations * 2
        assert features.shape[2] == expected_features

    def test_aggregate_bands_false(self, small_hsi):
        """Test feature extraction without band aggregation."""
        # Arrange: Create extractor without band aggregation
        extractor = GaborExtractor(aggregate_bands=False)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify non-aggregated features shape
        features = result["features"]
        assert features.shape[2] > 0
        assert features.ndim == 3

    def test_create_gabor_kernel(self, small_hsi):
        """Test Gabor kernel creation."""
        # Arrange: Create extractor
        extractor = GaborExtractor(sigma=3.0)
        frequency = 0.1
        theta = np.pi / 4

        # Act: Create kernel
        kernel = extractor._create_gabor_kernel(frequency, theta, 3.0)

        # Assert: Verify kernel properties
        assert kernel.ndim == 2
        assert kernel.shape[0] == kernel.shape[1]
        assert kernel.shape[0] % 2 == 1  # odd size
        assert np.isclose(kernel.sum(), 0.0, atol=1e-6)  # normalized

    def test_apply_gabor_filter(self, small_hsi):
        """Test Gabor filter application."""
        # Arrange: Create extractor and get a band
        extractor = GaborExtractor()
        band = small_hsi.reflectance[:, :, 0]
        frequency = 0.1
        theta = 0

        # Act: Apply filter
        magnitude, phase = extractor._apply_gabor_filter(band, frequency, theta)

        # Assert: Verify outputs
        assert magnitude.shape == band.shape
        assert phase.shape == band.shape
        assert magnitude.dtype == np.float64 or magnitude.dtype == np.float32
        assert phase.dtype == np.float64 or phase.dtype == np.float32

    def test_extract_gabor_features_single_band(self, small_hsi):
        """Test feature extraction from single band."""
        # Arrange: Create extractor and get a band
        extractor = GaborExtractor(frequencies=[0.1], thetas=[0])
        band = small_hsi.reflectance[:, :, 0]

        # Act: Extract features
        features = extractor._extract_gabor_features_single_band(band)

        # Assert: Verify output shape
        assert features.shape[0] == band.shape[0]
        assert features.shape[1] == band.shape[1]
        assert features.shape[2] == 2  # magnitude and energy

    @pytest.mark.parametrize("sigma", [2.0, 3.0, 5.0])
    def test_different_sigma_values(self, small_hsi, sigma):
        """Test extraction with different sigma values."""
        # Arrange: Create extractor with specific sigma
        extractor = GaborExtractor(sigma=sigma)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["features"].shape[0] == small_hsi.height

    @pytest.mark.parametrize(
        "frequencies,thetas",
        [
            ([0.05], [0]),
            ([0.1, 0.2], [0, np.pi / 2]),
            ([0.05, 0.1, 0.2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
        ],
    )
    def test_frequency_theta_combinations(self, small_hsi, frequencies, thetas):
        """Test extraction with different frequency and theta combinations."""
        # Arrange: Create extractor with specific parameters
        extractor = GaborExtractor(frequencies=frequencies, thetas=thetas)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = GaborExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "gabor"

    def test_kernel_size_calculation(self, small_hsi):
        """Test that kernel size is calculated correctly based on sigma."""
        # Arrange: Create extractor with specific sigma
        sigma = 4.0
        extractor = GaborExtractor(sigma=sigma)

        # Act: Create kernel
        kernel = extractor._create_gabor_kernel(0.1, 0, sigma)

        # Assert: Verify kernel size is odd and based on sigma
        expected_size = int(4 * sigma + 1)
        if expected_size % 2 == 0:
            expected_size += 1
        assert kernel.shape[0] == expected_size

    def test_kernel_size_even_adjustment(self, small_hsi):
        """Test that even kernel sizes are adjusted to odd."""
        # Arrange: Create extractor with sigma that produces even kernel size
        # sigma = 2.25 produces kernel_size = 10, which should be adjusted to 11
        sigma = 2.25
        extractor = GaborExtractor(sigma=sigma)

        # Act: Create kernel
        kernel = extractor._create_gabor_kernel(0.1, 0, sigma)

        # Assert: Verify kernel size is odd
        assert kernel.shape[0] % 2 == 1
        assert kernel.shape[0] == 11  # 10 + 1

    def test_nan_values_in_masked_regions(self, small_hsi):
        """Test that masked regions contain NaN values."""
        # Arrange: Create extractor
        extractor = GaborExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify NaN values in masked regions
        features = result["features"]
        mask = small_hsi.mask
        if not mask.all():
            assert np.isnan(features[~mask]).any()
