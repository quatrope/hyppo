"""Morphological Profile (MP) feature extractor for hyperspectral images."""

import numpy as np
from skimage.morphology import (
    closing,
    diamond,
    dilation,
    disk,
    erosion,
    footprint_rectangle,
    opening,
    reconstruction as morph_reconstruction,
)
from sklearn.decomposition import PCA
from hyppo.core import HSI
from .base import Extractor
from ._validators import (
    validate_all_in_set,
    validate_non_empty_list,
    validate_positive_int,
    validate_positive_int_list,
    validate_sufficient_bands,
)


class MPExtractor(Extractor):
    """
    Morphological Profile (MP) feature extractor for hyperspectral images.

    Implements morphological profiles based on differently shaped structuring
    elements as proposed by Lv et al. (2014). For each principal component,
    applies opening and closing operations using multiple SE shapes at multiple
    scales, creating a rich feature representation that captures structural
    information at different scales and orientations.

    Parameters
    ----------
    n_components : int, default=3
        Number of PCA components to retain before computing MPs.
    radii : list of int, default=[2, 4, 6, 8]
        Radii (or sizes) of structuring elements to apply. Each radius creates
        one opening and one closing operation per shape.
    shapes : list of str, default=['disk', 'square', 'diamond']
        List of structuring element shapes to use. Supported shapes:
        'disk', 'square', 'diamond', 'line'.
    use_reconstruction : bool, default=False
        If True, use opening/closing by reconstruction (preserves edges better).
        If False, use standard opening/closing operations.

    References
    ----------
    Lv, Z. Y., Zhang, P., Benediktsson, J. A., & Shi, W. Z. (2014).
    Morphological Profiles Based on Differently Shaped Structuring Elements
    for Classification of Images With Very High Spatial Resolution.
    IEEE Journal of Selected Topics in Applied Earth Observations and
    Remote Sensing, 7(12), 4644–4652. doi:10.1109/JSTARS.2014.2328618
    """

    def __init__(
        self,
        n_components=3,
        radii=[2, 4, 6, 8],
        shapes=["disk", "square", "diamond"],
        use_reconstruction=False,
    ):
        """Initialize MP extractor with morphological operation parameters."""
        super().__init__()
        self.n_components = n_components
        self.radii = sorted(radii)
        self.shapes = shapes
        self.use_reconstruction = use_reconstruction

    def _get_structuring_element(self, shape, radius):
        """Returns the structuring element based on shape and radius."""
        if shape == "disk":
            return disk(radius)
        elif shape == "square":
            return footprint_rectangle((2 * radius + 1, 2 * radius + 1))
        elif shape == "diamond":
            return diamond(radius)
        elif shape == "line":
            # Horizontal line SE
            return footprint_rectangle((1, 2 * radius + 1))
        else:
            raise ValueError(
                f"Unsupported shape '{shape}'. "
                f"Choose from: 'disk', 'square', 'diamond', 'line'."
            )

    def _opening_by_reconstruction(self, img, se):
        """Opening by reconstruction (geodesic opening). Removes bright
        structures smaller than SE while preserving edges of remaining
        structures better than standard opening."""
        # Erosion followed by dilation reconstruction
        eroded = erosion(img, se)
        return morph_reconstruction(eroded, img, method="dilation")

    def _closing_by_reconstruction(self, img, se):
        """Closing by reconstruction (geodesic closing).
        Fills dark structures smaller than SE while preserving edges."""
        # Dilation followed by erosion reconstruction
        dilated = dilation(img, se)
        return morph_reconstruction(dilated, img, method="erosion")

    def _compute_morphological_profile(self, image, shape):
        """
        Compute morphological profile for a single image and shape.

        The profile is structured as:
        [Opening_n, ..., Opening_1, Original, Closing_1, ..., Closing_n]

        Parameters
        ----------
        image : np.ndarray, shape (H, W)
            Input grayscale image (e.g., one principal component).
        shape : str
            Structuring element shape to use.

        Returns
        -------
        profile : np.ndarray, shape (H, W, 2*n_radii + 1)
            Morphological profile with openings, original, and closings.
        """
        h, w = image.shape
        n_radii = len(self.radii)
        profile_length = 2 * n_radii + 1

        profile = np.zeros((h, w, profile_length), dtype=np.float32)

        # Apply opening operations (from largest to smallest radius)
        # Store in reverse order: [Opening_n, ..., Opening_1]
        for i, radius in enumerate(reversed(self.radii)):
            se = self._get_structuring_element(shape, radius)

            if self.use_reconstruction:
                opened = self._opening_by_reconstruction(image, se)
            else:
                opened = opening(image, se)

            profile[:, :, i] = opened

        # Original image at center
        profile[:, :, n_radii] = image

        # Apply closing operations (from smallest to largest radius)
        # [Closing_1, ..., Closing_n]
        for i, radius in enumerate(self.radii):
            se = self._get_structuring_element(shape, radius)

            if self.use_reconstruction:
                closed = self._closing_by_reconstruction(image, se)
            else:
                closed = closing(image, se)

            profile[:, :, n_radii + 1 + i] = closed

        return profile

    def _extract(self, data: HSI, **inputs):
        """
        Extract morphological profile features from hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:
                - "features": np.ndarray, shape (H, W, n_features)
                    Stacked morphological profiles from all shapes and PCs.
                - "explained_variance_ratio": array
                    Variance ratio explained by each PCA component.
                - "n_components": int
                    Number of PCA components used.
                - "shapes": list of str
                    Shapes of structuring elements used.
                - "radii": list of int
                    Radii of structuring elements used.
                - "n_features": int
                    Total number of features extracted.
                - "use_reconstruction": bool
                    Whether reconstruction was used.
                Original spatial shape of the image (H, W).
        """
        reflectance = data.reflectance
        height, width, bands = reflectance.shape
        reflectance_reshaped = reflectance.reshape(-1, bands)

        pca = PCA(n_components=self.n_components)
        pcs = pca.fit_transform(reflectance_reshaped)
        pcs = pcs.reshape(height, width, self.n_components)

        # Extract morphological profiles for each PC and shape
        all_profiles = []

        # construccion del perfil
        for i in range(self.n_components):
            pc_img = pcs[:, :, i]

            for shape in self.shapes:
                # Compute profile for this PC and shape
                profile = self._compute_morphological_profile(pc_img, shape)
                all_profiles.append(profile)

        # Concatenate all profiles along feature dimension
        features = np.concatenate(all_profiles, axis=-1)

        # Calculate total number of features
        n_features_per_profile = 2 * len(self.radii) + 1
        n_total_features = (
            self.n_components * len(self.shapes) * n_features_per_profile
        )

        return {
            "features": features,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "n_components": self.n_components,
            "shapes": self.shapes,
            "radii": self.radii,
            "n_features": n_total_features,
            "use_reconstruction": self.use_reconstruction,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        validate_positive_int(self.n_components, "n_components")
        validate_non_empty_list(self.radii, "radii")
        validate_positive_int_list(self.radii, "radii")
        validate_non_empty_list(self.shapes, "shapes")
        validate_all_in_set(
            self.shapes,
            {"disk", "square", "diamond", "line"},
            "shape",
        )
        validate_sufficient_bands(data, self.n_components)
