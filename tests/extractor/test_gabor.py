"""Tests for GaborExtractor."""

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.gabor import GaborExtractor


class TestGaborExtractor:
    """Test cases for GaborExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 5x5x3 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 3).astype(np.float32)
        wavelengths = np.array([500.0, 600.0, 700.0])
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression_aggregate(self, regression_hsi):
        """Regression test: aggregated Gabor output must not change."""
        # Arrange
        extractor = GaborExtractor(
            frequencies=[0.1],
            thetas=[0],
            sigma=2.0,
            aggregate_bands=True,
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: magnitude channel
        expected_mag = np.array(
            [
                [0.03452614, 0.01580214, 0.01378948, 0.03779187, 0.06052897],
                [0.01190707, 0.01840728, 0.01215288, 0.03013321, 0.03749559],
                [0.03887709, 0.03123867, 0.00349533, 0.01892294, 0.03669482],
                [0.03999638, 0.02957242, 0.0183285, 0.03695732, 0.0643043],
                [0.04014437, 0.03036051, 0.03252354, 0.03821342, 0.05668537],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result["features"][:, :, 0], expected_mag, rtol=1e-5
        )

    def test_regression_non_aggregate(self, regression_hsi):
        """Regression test: non-aggregated output must not change."""
        # Arrange
        extractor = GaborExtractor(
            frequencies=[0.1],
            thetas=[0],
            sigma=2.0,
            aggregate_bands=False,
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: first band magnitude
        expected_band0_mag = np.array(
            [
                [0.07486316, 0.03604601, 0.01936384, 0.03977295, 0.0602481],
                [0.0063432, 0.01912639, 0.00610931, 0.02293983, 0.01873096],
                [0.04274749, 0.0487608, 0.00548443, 0.01778967, 0.04337907],
                [0.02857731, 0.03567681, 0.01647201, 0.02737262, 0.05940717],
                [0.01565858, 0.02815348, 0.00658716, 0.0070907, 0.0340844],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result["features"][:, :, 0], expected_band0_mag, rtol=1e-5
        )

    def test_kernel_mean_centered_and_normalized(self):
        """Test kernel is mean-centered and L1-normalized."""
        # Arrange
        extractor = GaborExtractor()
        kernel = extractor._create_gabor_kernel(0.1, np.pi / 4, 3.0)

        # Assert
        assert np.isclose(kernel.mean(), 0.0, atol=1e-7)
        assert np.isclose(np.abs(kernel).sum(), 1.0, atol=1e-6)

    def test_kernel_size_odd(self):
        """Test kernel size is always odd and based on sigma."""
        extractor = GaborExtractor()

        # sigma=3 → 4*3+1=13 (odd)
        k1 = extractor._create_gabor_kernel(0.1, 0, 3.0)
        assert k1.shape == (13, 13)

        # sigma=2.25 → 4*2.25+1=10 (even) → 11
        k2 = extractor._create_gabor_kernel(0.1, 0, 2.25)
        assert k2.shape == (11, 11)

    def test_magnitude_non_negative(self):
        """Test magnitude response is always non-negative."""
        # Arrange
        rng = np.random.RandomState(0)
        band = rng.rand(10, 10).astype(np.float32)
        extractor = GaborExtractor()

        # Act
        magnitude, _ = extractor._apply_gabor_filter(band, 0.1, 0)

        # Assert
        assert np.all(magnitude >= 0)

    def test_phase_range(self):
        """Test phase response is in [-pi, pi]."""
        # Arrange
        rng = np.random.RandomState(0)
        band = rng.rand(10, 10).astype(np.float32)
        extractor = GaborExtractor()

        # Act
        _, phase = extractor._apply_gabor_filter(band, 0.1, np.pi / 4)

        # Assert
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

    def test_energy_equals_magnitude_squared_per_band(self):
        """Test that energy feature equals magnitude squared per band."""
        # Arrange
        rng = np.random.RandomState(0)
        band = rng.rand(8, 8).astype(np.float32)
        extractor = GaborExtractor(frequencies=[0.1], thetas=[0])

        # Act
        features = extractor._extract_gabor_features_single_band(band)

        # Assert: features[:,:,0]=magnitude, features[:,:,1]=magnitude^2
        mag = features[:, :, 0]
        energy = features[:, :, 1]
        np.testing.assert_allclose(energy, mag**2)

    def test_aggregate_bands_averages(self, regression_hsi):
        """Test that aggregate=True computes mean across bands."""
        # Arrange
        extractor = GaborExtractor(
            frequencies=[0.1],
            thetas=[0],
            sigma=2.0,
            aggregate_bands=True,
        )
        ext_no_agg = GaborExtractor(
            frequencies=[0.1],
            thetas=[0],
            sigma=2.0,
            aggregate_bands=False,
        )

        # Act
        agg = extractor.extract(regression_hsi)["features"]
        no_agg = ext_no_agg.extract(regression_hsi)["features"]

        # Assert: aggregated = mean of per-band features
        n_bands = regression_hsi.reflectance.shape[2]
        n_feat_per_band = 2  # 1 freq * 1 theta * 2
        manual_mean = np.mean(
            [
                no_agg[:, :, i * n_feat_per_band : (i + 1) * n_feat_per_band]
                for i in range(n_bands)
            ],
            axis=0,
        )
        np.testing.assert_allclose(agg, manual_mean, atol=1e-6)

    def test_concatenate_band_features(self):
        """Test _concatenate_band_features produces correct shape."""
        # Arrange
        extractor = GaborExtractor(frequencies=[0.1], thetas=[0])
        band_feat1 = np.ones((5, 5, 2), dtype=np.float32)
        band_feat2 = np.full((5, 5, 2), 2.0, dtype=np.float32)

        # Act
        result = extractor._concatenate_band_features(
            [band_feat1, band_feat2], 5, 5
        )

        # Assert
        assert result.shape == (5, 5, 4)
        np.testing.assert_allclose(result[:, :, :2], 1.0)
        np.testing.assert_allclose(result[:, :, 2:], 2.0)

    def test_nan_in_masked_regions(self):
        """Test that masked pixels are NaN."""
        # Arrange
        reflectance = np.random.RandomState(0).rand(5, 5, 3).astype(np.float32)
        wavelengths = np.array([500.0, 600.0, 700.0])
        mask = np.ones((5, 5), dtype=bool)
        mask[0, 0] = False
        mask[2, 3] = False
        hsi = HSI(
            reflectance=reflectance,
            wavelengths=wavelengths,
            mask=mask,
        )
        extractor = GaborExtractor(frequencies=[0.1], thetas=[0])

        # Act
        result = extractor.extract(hsi)

        # Assert
        features = result["features"]
        assert np.all(np.isnan(features[0, 0, :]))
        assert np.all(np.isnan(features[2, 3, :]))
        assert not np.any(np.isnan(features[1, 1, :]))

    def test_feature_count(self, regression_hsi):
        """Test correct number of features for given params."""
        # Arrange
        freqs = [0.05, 0.1]
        thetas = [0, np.pi / 2, np.pi / 4]
        extractor = GaborExtractor(
            frequencies=freqs,
            thetas=thetas,
            aggregate_bands=True,
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: 2 freqs * 3 thetas * 2 (mag+energy) = 12
        assert result["features"].shape[2] == 12

    def test_feature_count_non_aggregate(self, regression_hsi):
        """Test feature count without band aggregation."""
        # Arrange
        freqs = [0.1]
        thetas = [0]
        n_bands = regression_hsi.reflectance.shape[2]
        extractor = GaborExtractor(
            frequencies=freqs,
            thetas=thetas,
            aggregate_bands=False,
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: n_bands * 1 freq * 1 theta * 2 = n_bands * 2
        assert result["features"].shape[2] == n_bands * 2

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = GaborExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        features = result["features"]
        assert features.shape[:2] == (small_hsi.height, small_hsi.width)
        assert features.ndim == 3
        # 3 freqs * 4 thetas * 2 = 24
        assert features.shape[2] == 24

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom frequencies and orientations."""
        # Arrange
        extractor = GaborExtractor(
            frequencies=[0.1, 0.2],
            thetas=[0, np.pi / 2],
            sigma=2.0,
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["features"].shape[:2] == (
            small_hsi.height,
            small_hsi.width,
        )

    @pytest.mark.parametrize("sigma", [2.0, 3.0, 5.0])
    def test_different_sigma_values(self, small_hsi, sigma):
        """Test extraction with different sigma values."""
        # Arrange
        extractor = GaborExtractor(sigma=sigma)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["features"].shape[0] == small_hsi.height

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert GaborExtractor.feature_name() == "gabor"
