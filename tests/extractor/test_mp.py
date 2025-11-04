"""Tests for MPExtractor."""

import pytest

from hyppo.extractor.mp import MPExtractor


class TestMPExtractor:
    """Test cases for MPExtractor."""

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result(self, sample_hsi):
        """Test results match reference values from literature."""
        # TODO: Implement validation against reference paper results
        # Lv et al. (2014) - Morphological Profiles Based on Differently Shaped
        # Structuring Elements for Classification
        pass

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = MPExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "bands_used" in result
        assert "radii" in result
        assert "structuring_element" in result
        assert "n_features" in result
        assert "original_shape" in result

        # Assert: Verify default parameter values
        assert result["radii"] == [1, 3, 5]
        assert result["structuring_element"] == "disk"

        # Assert: Verify feature shape
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom parameters."""
        # Arrange: Create extractor with custom parameters
        bands = [0, 1]
        radii = [1, 2]
        structuring_element = "square"
        extractor = MPExtractor(
            bands=bands, radii=radii, structuring_element=structuring_element
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify custom parameters
        assert result["bands_used"] == bands
        assert result["radii"] == radii
        assert result["structuring_element"] == structuring_element

    def test_all_bands_default(self, small_hsi):
        """Test that default behavior processes all bands."""
        # Arrange: Create extractor without specifying bands
        extractor = MPExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify all bands used
        assert len(result["bands_used"]) == small_hsi.reflectance.shape[2]

    def test_specific_bands(self, small_hsi):
        """Test extraction with specific bands."""
        # Arrange: Create extractor with specific bands
        bands = [0, 2]
        extractor = MPExtractor(bands=bands)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct bands used
        assert result["bands_used"] == bands

    @pytest.mark.parametrize(
        "structuring_element", ["disk", "square", "octagon"]
    )
    def test_different_structuring_elements(
        self, small_hsi, structuring_element
    ):
        """Test extraction with different structuring elements."""
        # Arrange: Create extractor with specific structuring element
        extractor = MPExtractor(structuring_element=structuring_element)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct structuring element
        assert result["structuring_element"] == structuring_element

    def test_get_structuring_element_disk(self, small_hsi):
        """Test disk structuring element generation."""
        # Arrange: Create extractor with disk
        extractor = MPExtractor(structuring_element="disk")

        # Act: Get structuring element
        elem = extractor._get_structuring_element(3)

        # Assert: Verify disk structure
        assert elem.ndim == 2
        assert elem.shape[0] == elem.shape[1]

    def test_get_structuring_element_square(self, small_hsi):
        """Test square structuring element generation."""
        # Arrange: Create extractor with square
        extractor = MPExtractor(structuring_element="square")

        # Act: Get structuring element
        elem = extractor._get_structuring_element(3)

        # Assert: Verify square structure
        assert elem.ndim == 2
        assert elem.shape[0] == elem.shape[1]
        assert elem.shape[0] == 7  # 2*3 + 1

    def test_get_structuring_element_octagon(self, small_hsi):
        """Test octagon structuring element generation."""
        # Arrange: Create extractor with octagon
        extractor = MPExtractor(structuring_element="octagon")

        # Act: Get structuring element
        elem = extractor._get_structuring_element(3)

        # Assert: Verify octagon structure
        assert elem.ndim == 2

    def test_get_structuring_element_invalid(self, small_hsi):
        """Test invalid structuring element raises error."""
        # Arrange: Create extractor with invalid element
        extractor = MPExtractor(structuring_element="invalid")

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="Unknown structuring element"):
            extractor._get_structuring_element(3)

    @pytest.mark.parametrize("radii", [[1], [1, 2], [1, 3, 5]])
    def test_different_radii(self, small_hsi, radii):
        """Test extraction with different radii."""
        # Arrange: Create extractor with specific radii
        extractor = MPExtractor(radii=radii)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct radii used
        assert result["radii"] == radii

    def test_n_features_calculation(self, small_hsi):
        """Test that n_features is calculated correctly."""
        # Arrange: Create extractor with known configuration
        bands = [0, 1]
        radii = [1, 2]
        extractor = MPExtractor(bands=bands, radii=radii)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify n_features = bands * radii * 4 operations
        expected_features = len(bands) * len(radii) * 4
        assert result["n_features"] == expected_features

    def test_validate_invalid_bands_type(self, small_hsi):
        """Test validation fails with invalid bands type."""
        # Arrange: Create extractor with invalid bands
        extractor = MPExtractor(bands="invalid")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="bands must be None or a non-empty"
        ):
            extractor.extract(small_hsi)

    def test_validate_empty_bands(self, small_hsi):
        """Test validation fails with empty bands list."""
        # Arrange: Create extractor with empty bands
        extractor = MPExtractor(bands=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="bands must be None or a non-empty"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_radii_type(self, small_hsi):
        """Test validation fails with invalid radii type."""
        # Arrange: Create extractor with invalid radii
        extractor = MPExtractor(radii="invalid")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="radii must be a non-empty"):
            extractor.extract(small_hsi)

    def test_validate_empty_radii(self, small_hsi):
        """Test validation fails with empty radii list."""
        # Arrange: Create extractor with empty radii
        extractor = MPExtractor(radii=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="radii must be a non-empty"):
            extractor.extract(small_hsi)

    def test_validate_negative_radii(self, small_hsi):
        """Test validation fails with negative radii."""
        # Arrange: Create extractor with negative radii
        extractor = MPExtractor(radii=[-1, 2])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="All radii must be positive"):
            extractor.extract(small_hsi)

    def test_validate_zero_radii(self, small_hsi):
        """Test validation fails with zero radii."""
        # Arrange: Create extractor with zero radii
        extractor = MPExtractor(radii=[0, 1])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="All radii must be positive"):
            extractor.extract(small_hsi)

    def test_validate_invalid_structuring_element(self, small_hsi):
        """Test validation fails with invalid structuring element."""
        # Arrange: Create extractor with invalid structuring element
        extractor = MPExtractor(structuring_element="invalid")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="structuring_element must be one of"
        ):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = MPExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "m_p"

    def test_original_shape_preserved(self, small_hsi):
        """Test that original spatial shape is recorded."""
        # Arrange: Create extractor
        extractor = MPExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify original shape
        assert result["original_shape"] == (small_hsi.height, small_hsi.width)

    def test_morphological_operations_applied(self, small_hsi):
        """Test that all morphological operations are applied."""
        # Arrange: Create extractor with single band and radius
        extractor = MPExtractor(bands=[0], radii=[1])

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify 4 operations (opening, closing, dilation, erosion)
        assert result["n_features"] == 4
