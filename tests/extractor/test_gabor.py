"""Tests for GaborExtractor."""

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.gabor import GaborExtractor


class TestGaborExtractor:
    """Test cases for GaborExtractor."""

    @pytest.fixture
    def mock_hsi(self):
        """Small 10x10x3 HSI for general tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(10, 10, 3).astype(np.float32)
        wavelengths = np.array([500.0, 600.0, 700.0])
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_feature_name(self):
        """Test that feature name is correctly returned."""
        assert GaborExtractor.feature_name() == "gabor"

    def test_output_shape_unichrome(self, mock_hsi):
        """Verify output shape for standard unichrome extraction.

        With B=3 bands, M=2 scales, N=4 orientations:
          n_features = B * M * N = 3 * 2 * 4 = 24
        """
        n_scales = 2
        n_orientations = 4
        extractor = GaborExtractor(
            n_scales=n_scales,
            n_orientations=n_orientations,
            use_opponent=False,
        )
        result = extractor.extract(mock_hsi)
        features = result["features"]

        expected_features = 3 * n_scales * n_orientations  # 24
        assert features.shape == (10, 10, expected_features)
        assert result["n_unichrome"] == expected_features
        assert result["n_opponent"] == 0
        assert result["n_features"] == expected_features

    def test_output_shape_with_opponent(self, mock_hsi):
        """Verify output shape when opponent features are enabled.

        With B=3 bands, M=2 scales, N=4 orientations:
          n_unichrome = B * M * N          = 3 * 2 * 4 = 24
          n_opponent  = (B*(B-1)/2) * M * N = 3 * 2 * 4 = 24
          n_features  = 48
        """
        extractor = GaborExtractor(
            n_scales=2, n_orientations=4, use_opponent=True
        )
        result = extractor.extract(mock_hsi)

        assert result["n_unichrome"] == 24
        assert result["n_opponent"] == 24
        assert result["n_features"] == 48
        assert result["features"].shape == (10, 10, 48)

    def test_kernel_size(self):
        """Kernels must be odd-sized and match expected size for given sigma.

        size = 2 * ceil(4 * sigma) + 1

        S0: sigma_sq=1.97 -> sigma~1.403 -> half=ceil(5.613)=6 -> size=13
        S1: sigma_sq=7.89 -> sigma~2.809 -> half=ceil(11.24)=12 -> size=25
        """
        extractor = GaborExtractor()

        k0 = extractor._make_kernel(1.97, 0.5, 0)
        assert k0.shape[0] % 2 == 1, "Kernel size must be odd"
        assert k0.shape == (13, 13)

        k1 = extractor._make_kernel(7.89, 0.25, 0)
        assert k1.shape[0] % 2 == 1, "Kernel size must be odd"
        assert k1.shape == (25, 25)

    def test_opponent_zero_for_identical_bands(self):
        """Opponent features must be exactly zero when bands are identical.

        d^ij_mn = h^i_mn - h^j_mn = 0 when I_i == I_j  (Rajadell Eq. 4)
        """
        band = np.random.RandomState(0).rand(10, 10).astype(np.float32)
        reflectance = np.stack([band, band], axis=-1)
        hsi = HSI(
            reflectance=reflectance, wavelengths=np.array([500.0, 600.0])
        )

        extractor = GaborExtractor(
            n_scales=1, n_orientations=1, use_opponent=True
        )
        result = extractor.extract(hsi)

        n_uni = result["n_unichrome"]
        opponent_part = result["features"][:, :, n_uni:]
        np.testing.assert_allclose(opponent_part, 0.0, atol=1e-6)

    def test_validate_opponent_band_limit(self):
        """Should raise ValueError if use_opponent=True and B > 10."""
        ref_11 = np.random.rand(5, 5, 11).astype(np.float32)
        hsi_11 = HSI(reflectance=ref_11, wavelengths=np.arange(11))

        extractor = GaborExtractor(use_opponent=True)

        with pytest.raises(ValueError, match="Apply band selection upstream"):
            extractor.extract(hsi_11)

    def test_custom_sigmas_and_frequencies(self, mock_hsi):
        """Custom sigmas_sq and frequencies must be used in the filter bank."""
        custom_sigmas = [2.0, 4.0]
        custom_freqs = [0.1, 0.05]
        extractor = GaborExtractor(
            n_scales=2,
            sigmas_sq=custom_sigmas,
            frequencies=custom_freqs,
        )
        result = extractor.extract(mock_hsi)
        assert result["scales"] == 2

        kernels, params = extractor._build_filter_bank()

        # Scale 0: params[0..3] all share sigma_sq=2.0, freq=0.1
        assert params[0][0] == 2.0, "S0 sigma_sq should be 2.0"
        assert params[0][1] == 0.1, "S0 freq should be 0.1"

        # Scale 1: params[4..7] all share sigma_sq=4.0, freq=0.05
        n_ori = extractor.n_orientations  # 4
        assert params[n_ori][0] == 4.0, "S1 sigma_sq should be 4.0"
        assert params[n_ori][1] == 0.05, "S1 freq should be 0.05"

    def test_sigmas_length_mismatch_raises(self, mock_hsi):
        """Sigmas_sq length != n_scales must raise ValueError on extract."""
        extractor = GaborExtractor(n_scales=2, sigmas_sq=[1.0])  # wrong length
        with pytest.raises(ValueError, match="sigmas_sq has"):
            extractor.extract(mock_hsi)

    def test_frequencies_length_mismatch_raises(self, mock_hsi):
        """Frequencies length != n_scales must raise ValueError on extract."""
        extractor = GaborExtractor(
            n_scales=2, frequencies=[0.5, 0.25, 0.1]
        )  # wrong length
        with pytest.raises(ValueError, match="frequencies has"):
            extractor.extract(mock_hsi)

    def test_feature_layout_order(self):
        """Unichrome features are ordered: all filters for band 0, then band 1.

        With n_scales=1, n_orientations=4:
          idx 0-3: B0 filters (0°, 45°, 90°, 135°)
          idx 4-7: B1 filters (0°, 45°, 90°, 135°)

        Band 0 = horizontal grating -> filter 0° should respond strongly.
        Band 1 = vertical grating   -> filter 90° should respond strongly.
        The 0° response of B0 must exceed the 0° response of B1.
        """
        size = 50
        y, x = np.mgrid[0:size, 0:size]
        band_h = np.sin(2 * np.pi * 0.45 * y).astype(np.float32)  # horizontal
        band_v = np.sin(2 * np.pi * 0.45 * x).astype(np.float32)  # vertical
        reflectance = np.stack([band_h, band_v], axis=-1)
        hsi = HSI(
            reflectance=reflectance, wavelengths=np.array([500.0, 600.0])
        )

        extractor = GaborExtractor(
            n_scales=1,
            n_orientations=4,
            sigmas_sq=[1.97],
            frequencies=[0.45],
        )
        result = extractor.extract(hsi)
        features = result["features"]

        # idx 0: B0 filtered at 0° (horizontal filter -> responds to band_h)
        # idx 4: B1 filtered at 0° (horizontal filter -> weak vs band_v)
        resp_b0_0deg = np.abs(features[:, :, 0]).mean()
        resp_b1_0deg = np.abs(features[:, :, 4]).mean()
        assert resp_b0_0deg > resp_b1_0deg, (
            "B0 (horizontal grating) should produce stronger response "
            "than B1 (vertical grating) when filtered at 0°"
        )

        # Symmetrically: 90° filter responds more to B1 (vertical) than B0
        # idx 2: B0 at 90°,  idx 6: B1 at 90°
        resp_b0_90deg = np.abs(features[:, :, 2]).mean()
        resp_b1_90deg = np.abs(features[:, :, 6]).mean()
        assert resp_b1_90deg > resp_b0_90deg, (
            "B1 (vertical grating) should produce stronger response "
            "than B0 (horizontal grating) when filtered at 90°"
        )
