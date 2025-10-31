"""Local Binary Pattern (LBP) texture feature extractor for hyperspectral images."""

import numpy as np
from skimage.feature import local_binary_pattern

from hyppo.core import HSI
from .base import Extractor


class LBPExtractor(Extractor):
    """
    Local Binary Pattern (LBP) feature extractor for hyperspectral images.

    Computes LBP texture features for specified bands of an HSI cube. By default,
    all bands are processed. Each pixel's spectral neighborhood is encoded using
    a circular pattern of points.

    Parameters
    ----------
    bands : list of int or None
        List of band indices to process. If None, all bands are used.
    radius : float
        Radius of the circle of sampling points (default: 3).
    n_points : int or None
        Number of sampling points for the LBP. If None, defaults to 8 * radius.
    method : str
        LBP computation method. Options: "default", "ror", "uniform", "nri_uniform", "var".
        Default is "uniform".

    References
    ----------
    Ojala, T., & Pietikainen, M. (2002). Multiresolution gray-scale and rotation invariant
    texture classification with local binary patterns. IEEE Transactions on Pattern Analysis
    and Machine Intelligence, 24(7), 971–987. doi:10.1109/TPAMI.2002.1017623

    """

    def __init__(self, bands=None, radius=3, n_points=None, method="uniform"):
        """Initialize LBP extractor with texture parameters."""
        super().__init__()

        valid_methods = ["default", "ror", "uniform", "nri_uniform", "var"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        self.bands = bands
        self.radius = radius
        self.n_points = n_points if n_points is not None else 8 * radius
        self.method = method

    def _compute_lbp_responses(self, reflectance):
        """Compute LBP for all specified bands"""
        responses = {}
        if self.bands is None:
            bands_to_process = list(range(reflectance.shape[2]))
        else:
            bands_to_process = self.bands

            # Validate band indices
            max_band_index = reflectance.shape[2] - 1
            for band in bands_to_process:
                if band < 0 or band > max_band_index:
                    raise ValueError(
                        f"Band index {band} is out of range for input with {max_band_index + 1} bands."
                    )

        for band_idx in bands_to_process:
            band = reflectance[:, :, band_idx]
            # Normalize band to [0, 1] range for better LBP computation
            norm_band = self._normalize_band(band)

            # Compute LBP for this band
            lbp = local_binary_pattern(
                norm_band, P=self.n_points, R=self.radius, method=self.method
            )
            responses[band_idx] = lbp

        return responses, bands_to_process

    def _normalize_band(self, band):
        """Normalize band values to uint8 in [0, 255] range."""
        band_min, band_max = band.min(), band.max()

        if band_max == band_min:
            return np.zeros_like(band, dtype=np.uint8)

        normalized = (band - band_min) / (band_max - band_min)
        scaled = (normalized * 255).astype(np.uint8)
        return scaled

    def _extract(self, data: HSI, **inputs):
        """
        Extract LBP features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object.

        Returns
        -------
        dict
            Dictionary containing:
            - "features": LBP feature cube (H, W, n_features)
            - "bands_used": List of bands processed
            - "radius": Radius used for LBP
            - "n_points": Number of points used for LBP
            - "method": LBP computation method
            - "original_shape": Original (H, W) shape of HSI
            - "n_features": Number of LBP features extracted per pixel
        """
        reflectance = data.reflectance  # Hyperspectral cube: shape (H, W, B)
        height, width, bands = reflectance.shape
        original_shape = (height, width)

        # Compute LBP responses for specified bands
        responses, bands_used = self._compute_lbp_responses(reflectance)

        # Extract features from precomputed responses
        features = np.stack([responses[band] for band in bands_used], axis=-1)

        return {
            "features": features,
            "bands_used": bands_used,
            "radius": self.radius,
            "n_points": self.n_points,
            "method": self.method,
            "original_shape": original_shape,
            "n_features": features.shape[-1],
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.bands is not None and (
            not isinstance(self.bands, list) or not self.bands
        ):
            raise ValueError(
                "bands must be None or a non-empty list of integers."
            )

        if not isinstance(self.radius, (int, float)) or self.radius <= 0:
            raise ValueError("radius must be a positive number.")

        if not isinstance(self.n_points, int) or self.n_points <= 0:
            raise ValueError("n_points must be a positive integer.")

        if self.n_points < 3:
            raise ValueError(
                "n_points must be at least 3 for meaningful LBP computation."
            )
