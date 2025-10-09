"""Tests for GLCMExtractor."""

import pytest
import numpy as np
from hyppo.extractor.glcm import GLCMExtractor


class TestGLCMExtractor:
    """Test cases for GLCMExtractor."""

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = GLCMExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "bands_used" in result
        assert "distances" in result
        assert "angles" in result
        assert "properties" in result
        assert "levels_used" in result
        assert "window_sizes" in result
        assert "orientation_mode" in result
        assert "n_features_per_scale" in result
        assert "n_features_per_band" in result
        assert "total_features" in result
        assert "original_shape" in result

        # Assert: Verify default parameters
        assert result["distances"] == [1]
        assert result["properties"] == ["contrast", "entropy", "correlation", "dissimilarity"]
        assert result["window_sizes"] == [7]
        assert result["orientation_mode"] == "separate"

        # Assert: Verify feature shape
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom parameters."""
        # Arrange: Create extractor with custom parameters
        bands = [0, 1]
        distances = [1, 2]
        angles = [0, np.pi / 2]
        properties = ["contrast", "correlation"]
        window_sizes = [5]
        extractor = GLCMExtractor(
            bands=bands,
            distances=distances,
            angles=angles,
            properties=properties,
            window_sizes=window_sizes,
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify custom parameters
        assert result["bands_used"] == bands
        assert result["distances"] == distances
        assert result["properties"] == properties
        assert result["window_sizes"] == window_sizes

    def test_auto_determine_levels(self, small_hsi):
        """Test automatic level determination."""
        # Arrange: Create extractor without explicit levels
        extractor = GLCMExtractor(levels=None)
        image = small_hsi.reflectance[:, :, 0]

        # Act: Determine levels
        levels = extractor._auto_determine_levels(image)

        # Assert: Verify levels are within valid range
        assert levels >= 32
        assert levels <= 64

    def test_auto_determine_levels_low_dynamic_range(self, small_hsi):
        """Test level determination for low dynamic range."""
        # Arrange: Create extractor and low dynamic range image
        extractor = GLCMExtractor(levels=None)
        image = np.random.rand(10, 10) * 20  # range 0-20

        # Act: Determine levels
        levels = extractor._auto_determine_levels(image)

        # Assert: Verify levels = 32 for low dynamic range
        assert levels == 32

    def test_auto_determine_levels_medium_dynamic_range(self, small_hsi):
        """Test level determination for medium dynamic range."""
        # Arrange: Create extractor and medium dynamic range image
        extractor = GLCMExtractor(levels=None)
        image = np.random.rand(10, 10) * 50  # range 0-50

        # Act: Determine levels
        levels = extractor._auto_determine_levels(image)

        # Assert: Verify levels <= 64
        assert levels <= 64

    def test_auto_determine_levels_high_dynamic_range(self, small_hsi):
        """Test level determination for high dynamic range."""
        # Arrange: Create extractor and high dynamic range image
        extractor = GLCMExtractor(levels=None)
        image = np.random.rand(10, 10) * 100  # range 0-100

        # Act: Determine levels
        levels = extractor._auto_determine_levels(image)

        # Assert: Verify levels = 64 for high dynamic range
        assert levels == 64

    def test_normalize_to_levels(self, small_hsi):
        """Test band normalization to levels."""
        # Arrange: Create extractor
        extractor = GLCMExtractor()
        band = small_hsi.reflectance[:, :, 0]
        levels = 32

        # Act: Normalize band
        normalized = extractor._normalize_to_levels(band, levels)

        # Assert: Verify normalization
        assert normalized.dtype == np.uint8
        assert normalized.min() >= 0
        assert normalized.max() <= levels - 1

    def test_normalize_to_levels_constant_band(self, small_hsi):
        """Test normalization of constant band."""
        # Arrange: Create extractor and constant band
        extractor = GLCMExtractor()
        band = np.ones((10, 10)) * 5.0
        levels = 32

        # Act: Normalize band
        normalized = extractor._normalize_to_levels(band, levels)

        # Assert: Verify all zeros for constant band
        assert np.all(normalized == 0)

    def test_extract_glcm_multiscale(self, small_hsi):
        """Test multiscale GLCM extraction."""
        # Arrange: Create extractor
        extractor = GLCMExtractor(window_sizes=[3, 5])
        image = small_hsi.reflectance[:, :, 0]

        # Act: Extract multiscale features
        features, levels = extractor._extract_glcm_multiscale(image)

        # Assert: Verify output shape
        assert features.shape[0] == image.shape[0]
        assert features.shape[1] == image.shape[1]
        assert features.ndim == 3
        assert levels > 0

    @pytest.mark.parametrize("orientation_mode", ["separate", "look_direction", "average"])
    def test_different_orientation_modes(self, small_hsi, orientation_mode):
        """Test extraction with different orientation modes."""
        # Arrange: Create extractor with specific orientation mode
        extractor = GLCMExtractor(orientation_mode=orientation_mode)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct orientation mode
        assert result["orientation_mode"] == orientation_mode
        assert "features" in result

    def test_validate_invalid_bands(self, small_hsi):
        """Test validation fails with invalid bands."""
        # Arrange: Create extractor with invalid bands
        extractor = GLCMExtractor(bands="invalid") # type: ignore

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="bands must be None or a non-empty list"):
            extractor.extract(small_hsi)

    def test_validate_invalid_distances(self, small_hsi):
        """Test validation fails with invalid distances."""
        # Arrange: Create extractor with invalid distances
        extractor = GLCMExtractor(distances=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="distances must be a non-empty list"):
            extractor.extract(small_hsi)

    def test_validate_invalid_angles(self, small_hsi):
        """Test validation fails with invalid angles."""
        # Arrange: Create extractor with invalid angles
        extractor = GLCMExtractor(angles=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="angles must be a non-empty list"):
            extractor.extract(small_hsi)

    def test_validate_invalid_window_sizes_empty(self, small_hsi):
        """Test validation fails with empty window_sizes."""
        # Arrange: Create extractor with empty window_sizes
        extractor = GLCMExtractor(window_sizes=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="window_sizes must be a non-empty list"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("invalid_window", [2, 4, 1])
    def test_validate_invalid_window_size_values(self, small_hsi, invalid_window):
        """Test validation fails with invalid window size values."""
        # Arrange: Create extractor with invalid window size
        extractor = GLCMExtractor(window_sizes=[invalid_window])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="Each window size must be an odd integer"):
            extractor.extract(small_hsi)

    def test_validate_invalid_levels(self, small_hsi):
        """Test validation fails with invalid levels."""
        # Arrange: Create extractor with invalid levels
        extractor = GLCMExtractor(levels=1)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="levels must be greater than 1"):
            extractor.extract(small_hsi)

    def test_validate_invalid_properties(self, small_hsi):
        """Test validation fails with invalid properties."""
        # Arrange: Create extractor with empty properties
        extractor = GLCMExtractor(properties=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="properties must be a non-empty list"):
            extractor.extract(small_hsi)

    def test_validate_invalid_orientation_mode(self, small_hsi):
        """Test validation fails with invalid orientation mode."""
        # Arrange: Create extractor with invalid orientation mode
        extractor = GLCMExtractor(orientation_mode="invalid")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="orientation_mode must be"):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = GLCMExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "g_l_c_m"

    def test_symmetric_parameter(self, small_hsi):
        """Test extraction with symmetric parameter."""
        # Arrange: Create extractor with symmetric=False
        extractor = GLCMExtractor(symmetric=False)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result

    def test_explicit_levels(self, small_hsi):
        """Test extraction with explicit level specification."""
        # Arrange: Create extractor with explicit levels
        extractor = GLCMExtractor(levels=32)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify levels used
        assert all(level == 32 for level in result["levels_used"])

    def test_multiple_bands(self, small_hsi):
        """Test extraction with multiple specific bands."""
        # Arrange: Create extractor with specific bands
        bands = [0, 1, 2]
        extractor = GLCMExtractor(bands=bands)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct bands used
        assert result["bands_used"] == bands

    def test_all_bands_default(self, small_hsi):
        """Test that default behavior processes all bands."""
        # Arrange: Create extractor without specifying bands
        extractor = GLCMExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify all bands used
        assert len(result["bands_used"]) == small_hsi.reflectance.shape[2]

    def test_compute_glcm_features_optimized(self, small_hsi):
        """Test optimized GLCM computation."""
        # Arrange: Create extractor and patches
        extractor = GLCMExtractor()
        patches = np.random.randint(0, 32, size=(5, 7, 7), dtype=np.uint8)
        levels = 32

        # Act: Compute features
        features = extractor._compute_glcm_features_optimized(patches, levels)

        # Assert: Verify features shape
        assert features.shape[0] == 5
        assert features.shape[1] > 0
