"""Minimum Noise Fraction (MNF) feature extractor for hyperspectral images."""

import numpy as np
import warnings

from hyppo.core import HSI
from .base import Extractor


class MNFExtractor(Extractor):
    """
    Minimum Noise Fraction (MNF) feature extractor for hyperspectral images.

    MNF is a noise-adjusted transformation that orders components by decreasing
    signal-to-noise ratio (SNR), following Green et al. (1988). The algorithm
    consists of:
    1. Noise covariance estimation from spatial differences
    2. Noise whitening transformation
    3. PCA in the whitened space (ordered by SNR)

    Parameters
    ----------
    n_components : int, default=5
        Number of components to keep.

    References
    ----------
    .. [1] Green, A. A., Berman, M., Switzer, P., & Craig, M. D. (1988).
       A transformation for ordering multispectral data in terms of image
       quality with implications for noise removal.
       IEEE Transactions on Geoscience and Remote Sensing, 26(1), 65–74.
    """

    def __init__(self, n_components=5):
        """Initialize MNF extractor with transformation parameters."""
        super().__init__()
        self.n_components = n_components

    @classmethod
    def feature_name(cls) -> str:
        """Return the feature name."""
        return "mnf"

    def _extract(self, data: HSI, **inputs):
        """
        Extract MNF features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:

            - "features" : ndarray (H, W, n_components)
                MNF-transformed features
            - "n_features" : int
                Number of components extracted
            - "original_shape" : tuple
                Original HSI shape (H, W, bands)
            - "mean" : ndarray (bands,)
                Mean of each band
            - "noise_eigenvalues" : ndarray
                Eigenvalues of noise covariance
            - "snr_eigenvalues" : ndarray
                Eigenvalues in whitened space (signal+noise variance)
            - "snr_estimates" : ndarray
                Estimated SNR for each component (eigenvalue - 1)
            - "snr_ratio" : ndarray
                Proportion of total SNR per component
            - "cumulative_snr_ratio" : ndarray
                Cumulative SNR ratio
            - "whitening_matrix" : ndarray (bands, bands)
                Noise whitening transformation
            - "projection_matrix" : ndarray (bands, n_components)
                MNF projection vectors
        """
        X = data.reflectance
        h, w, bands = X.shape
        X_flat = X.reshape(-1, bands)

        # Center data
        X_mean = X_flat.mean(axis=0)
        X_centered = X_flat - X_mean

        # Noise covariance estimation (spatial differences)
        # Compute first-order differences in both spatial directions
        diffs_h = X[1:, :, :] - X[:-1, :, :]  # vertical differences
        diffs_w = X[:, 1:, :] - X[:, :-1, :]  # horizontal differences
        diffs = np.concatenate(
            [diffs_h.reshape(-1, bands), diffs_w.reshape(-1, bands)], axis=0
        )
        noise_cov = np.cov(diffs.T) / 2.0

        # Numerical stabilization: add small ridge to avoid singular matrix
        noise_cov += 1e-8 * np.eye(bands)

        # Noise whitening
        eigvals_n, eigvecs_n = np.linalg.eigh(noise_cov)

        # Check for negative eigenvalues (numerical issues)
        n_negative = np.sum(eigvals_n < 0)
        if n_negative > 0:
            warnings.warn(
                f"Found {n_negative} negative noise eigenvalues. "
                "Clipping to small positive value."
            )
        eigvals_n = np.clip(eigvals_n, 1e-12, None)

        # Whitening matrix: Σ_N^(-1/2)
        W = eigvecs_n @ np.diag(1.0 / np.sqrt(eigvals_n)) @ eigvecs_n.T

        # Apply whitening
        X_whitened = X_centered @ W.T

        # Signal covariance in whitened space
        Sigma = np.cov(X_whitened.T)
        eigvals_s, eigvecs_s = np.linalg.eigh(Sigma)

        # Sort by decreasing eigenvalue (highest SNR first)
        order = np.argsort(eigvals_s)[::-1]
        eigvals_s = eigvals_s[order]
        eigvecs_s = eigvecs_s[:, order]

        # MNF projection
        n_comp = min(self.n_components, bands)
        A = eigvecs_s[:, :n_comp]

        # Project data
        features_2d = X_whitened @ A
        features = features_2d.reshape(h, w, n_comp)

        # Compute SNR estimates (eigenvalue - 1, since noise variance = 1)
        snr_estimates = np.maximum(eigvals_s[:n_comp] - 1.0, 0)

        # Compute SNR ratio (similar to explained variance ratio)
        total_snr = np.sum(eigvals_s)
        snr_ratio = (
            eigvals_s[:n_comp] / total_snr
            if total_snr > 0
            else np.zeros(n_comp)
        )

        return {
            "features": features,
            "n_features": n_comp,
            "original_shape": (h, w, bands),
            "mean": X_mean,
            "noise_eigenvalues": eigvals_n,
            "snr_eigenvalues": eigvals_s[:n_comp],
            "snr_estimates": snr_estimates,
            "snr_ratio": snr_ratio,
            "cumulative_snr_ratio": np.cumsum(snr_ratio),
            "whitening_matrix": W,
            "projection_matrix": A,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        h, w, bands = data.reflectance.shape
        if h < 2 or w < 2:
            raise ValueError(
                "Image must have at least 2x2 spatial dimensions "
                "for spatial difference computation"
            )
