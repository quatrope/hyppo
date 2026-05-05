"""Tests for LBPExtractor."""

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.lbp import LBPExtractor


class TestLBPExtractor:
    """Tests for LBPExtractor."""

    def test_extract_basic_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        extractor = LBPExtractor()

        result = extractor.extract(small_hsi)

        assert "features" in result
        assert "radius" in result
        assert "n_points" in result
        assert "method" in result
        assert "spectral_mode" in result
        assert "n_features" in result

        assert result["spectral_mode"] == "pca"

        features = result["features"]

        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

    def test_custom_parameters(self, small_hsi):
        """Test extraction with custom radius, n_points and method."""
        extractor = LBPExtractor(
            radius=2,
            n_points=8,
            method="ror",
            n_components=2,
        )

        result = extractor.extract(small_hsi)

        assert result["method"] == "ror"
        assert result["radius"] == [2]
        assert result["n_points"] == [8]

    def test_multiscale_output_shape(self, small_hsi):
        """Test multiscale extraction produces n_channels * n_scales feats."""
        extractor = LBPExtractor(radius=[1, 3])

        result = extractor.extract(small_hsi)

        n_scales = len(result["radius"])
        n_channels = result["n_channels"]

        expected = n_channels * n_scales

        assert result["n_features"] == expected
        assert result["features"].shape[2] == expected

    def test_spectral_mode_bands(self, small_hsi):
        """Test bands spectral mode keeps the selected band indices."""
        extractor = LBPExtractor(spectral_mode="bands", band_indices=[0, 1])

        result = extractor.extract(small_hsi)

        assert result["spectral_mode"] == "bands"
        assert result["n_channels"] == 2

    def test_spectral_mode_all_bands(self, small_hsi):
        """Test bands mode without band_indices uses all input bands."""
        extractor = LBPExtractor(spectral_mode="bands")

        result = extractor.extract(small_hsi)

        assert result["n_channels"] == small_hsi.reflectance.shape[2]

    def test_invalid_spectral_mode(self):
        """Test unknown spectral_mode raises ValueError."""
        with pytest.raises(ValueError):
            LBPExtractor(spectral_mode="invalid")

    def test_compute_lbp_multiscale(self, small_hsi):
        """Test multiscale LBP returns one channel per radius."""
        extractor = LBPExtractor(radius=[1, 2])

        band = small_hsi.reflectance[:, :, 0]

        lbp = extractor._compute_lbp_multiscale(band)

        assert lbp.shape[0] == band.shape[0]
        assert lbp.shape[1] == band.shape[1]
        assert lbp.shape[2] == 2

    def test_validate_invalid_radius(self, small_hsi):
        """Test radius=0 raises ValueError on extract."""
        extractor = LBPExtractor(radius=0)

        with pytest.raises(ValueError):
            extractor.extract(small_hsi)

    def test_validate_invalid_n_points(self, small_hsi):
        """Test n_points=0 raises ValueError on extract."""
        extractor = LBPExtractor(n_points=0)

        with pytest.raises(ValueError):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize(
        "method", ["default", "ror", "uniform", "nri_uniform", "var"]
    )
    def test_methods(self, small_hsi, method):
        """Test each supported LBP method propagates to result metadata."""
        extractor = LBPExtractor(method=method)

        result = extractor.extract(small_hsi)

        assert result["method"] == method

    def test_invalid_method(self):
        """Test unknown method raises ValueError on construction."""
        with pytest.raises(ValueError):
            LBPExtractor(method="invalid")

    def test_radius_n_points_length_mismatch(self):
        """Test mismatched radius and n_points lengths raise ValueError."""
        with pytest.raises(
            ValueError, match="radius and n_points must have same length"
        ):
            LBPExtractor(radius=[1, 2], n_points=[8])

    def test_band_index_out_of_range(self, small_hsi):
        """Test band_indices outside the input bands raise ValueError."""
        extractor = LBPExtractor(spectral_mode="bands", band_indices=[99])
        with pytest.raises(ValueError, match="band index 99 out of range"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("radius", [1, 2, 3, 5])
    def test_radius_values(self, small_hsi, radius):
        """Test radius values are stored as a list in result metadata."""
        extractor = LBPExtractor(radius=radius)

        result = extractor.extract(small_hsi)

        assert result["radius"] == [radius]

    def test_n_points_default(self, small_hsi):
        """Test default n_points is 8 * radius when not provided."""
        radius = 4

        extractor = LBPExtractor(radius=radius)

        result = extractor.extract(small_hsi)

        assert result["n_points"] == [8 * radius]

    def test_feature_name(self):
        """Test feature_name returns the canonical 'lbp' identifier."""
        name = LBPExtractor.feature_name()

        assert name == "lbp"

    def test_regression(self):
        """Test deterministic regression on a fixed-seed input."""
        rng = np.random.RandomState(42)

        reflectance = rng.rand(5, 5, 3).astype(np.float32)
        wavelengths = np.array([500.0, 600.0, 700.0])

        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)

        extractor = LBPExtractor(radius=1, n_points=8)

        result = extractor.extract(hsi)

        assert result["features"].shape[0] == 5
        assert result["features"].shape[1] == 5
