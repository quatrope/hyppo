from .base import Extractor
from hyppo.core import HSI
import numpy as np
from skimage.morphology import (
    opening,
    closing,
    dilation,
    erosion,
    disk,
    square,
    octagon,
)


class MPExtractor(Extractor):
    """
    Morphological Profile (MP) feature extractor for hyperspectral images.

    Applies morphological operations (opening, closing, dilation, erosion)
    at multiple scales to selected bands, using differently shaped structuring
    elements. The choice of structuring element shape and radius affects the
    extracted features, in line with the methodology proposed by Lv et al. (2014).

    Parameters
    ----------
    bands : list of int or None, optional
        Bands to process. If None, all bands are used.
    radii : list of int, optional
        List of structuring element radii to apply. Default is [1, 3, 5].
    structuring_element : {'disk', 'square', 'octagon'}, optional
        Type of structuring element to use. Default is 'disk'.

    References
    ----------
    Lv, Z. Y., Zhang, P., Benediktsson, J. A., & Shi, W. Z. (2014).
    Morphological Profiles Based on Differently Shaped Structuring Elements
    for Classification of Images With Very High Spatial Resolution.
    IEEE Journal of Selected Topics in Applied Earth Observations and
    Remote Sensing, 7(12), 4644–4652. doi:10.1109/JSTARS.2014.2328618
    """

    def __init__(self, bands=None, radii=None, structuring_element="disk"):
        super().__init__()
        self.bands = bands
        self.radii = radii if radii is not None else [1, 3, 5]
        self.structuring_element = structuring_element

    def _get_structuring_element(self, r):
        """Return a structuring element of specified shape and radius."""
        if self.structuring_element == "disk":
            return disk(r)
        elif self.structuring_element == "square":
            return square(2 * r + 1)
        elif self.structuring_element == "octagon":
            return octagon(r, r)
        else:
            raise ValueError(f"Unknown structuring element: {self.structuring_element}")

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
            - features : ndarray of shape (H, W, n_features)
                Stacked morphological features for each band and scale.
            - bands_used : list of int
                Bands actually used in extraction.
            - radii : list of int
                Radii of structuring elements applied.
            - structuring_element : str
                Type of structuring element used.
            - n_features : int
                Total number of features extracted.
            - original_shape : tuple of int
                Original spatial shape of the image (H, W).
        """
        X = data.reflectance  # (H, W, B)
        h, w, b = X.shape

        bands_to_use = self.bands if self.bands is not None else range(b)

        features_list = []

        for band_idx in bands_to_use:
            band = X[:, :, band_idx]
            for r in self.radii:
                elem = self._get_structuring_element(r)
                # Opening
                opened = opening(band, elem)
                features_list.append(opened)
                # Closing
                closed = closing(band, elem)
                features_list.append(closed)
                # Dilation
                dilated = dilation(band, elem)
                features_list.append(dilated)
                # Erosion
                eroded = erosion(band, elem)
                features_list.append(eroded)
        # Stack features: shape (H, W, n_features)
        features = np.stack(features_list, axis=-1)

        return {
            "features": features,
            "bands_used": list(bands_to_use),
            "radii": self.radii,
            "structuring_element": self.structuring_element,
            "n_features": features.shape[-1],
            "original_shape": (h, w),
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.bands is not None and (
            not isinstance(self.bands, (list, tuple)) or not self.bands
        ):
            raise ValueError(
                "bands must be None or a non-empty list/tuple of integers."
            )
        if not isinstance(self.radii, (list, tuple)) or not self.radii:
            raise ValueError(
                "radii must be a non-empty list/tuple of positive integers."
            )
        if any(r <= 0 for r in self.radii):
            raise ValueError("All radii must be positive integers.")
        if self.structuring_element not in ("disk", "square", "octagon"):
            raise ValueError(
                "structuring_element must be one of 'disk', 'square', 'octagon'."
            )
