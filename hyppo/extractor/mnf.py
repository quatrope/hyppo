"""Minimum Noise Fraction (MNF) feature extractor for hyperspectral images."""

import numpy as np
from sklearn.decomposition import PCA

from hyppo.core import HSI
from .base import Extractor


class MNFExtractor(Extractor):
    """
    Minimum Noise Fraction (MNF) feature extractor for hyperspectral images.

    MNF is a noise-adjusted principal component analysis that maximizes
    signal-to-noise ratio by whitening the noise covariance before applying PCA.

    Parameters
    ----------
    n_components : int, default=5
        Number of components to keep.
    whiten : bool, default=False
        Whether to whiten the components.
    random_state : int, RandomState instance or None, default=42
        Random state for reproducibility.

    References
    ----------
    .. [1] Green, A. A., Berman, M., Switzer, P., & Craig, M. D. (1988).
       A transformation for ordering multispectral data in terms of image
       quality with implications for noise removal.
       IEEE Transactions on Geoscience and Remote Sensing, 26(1), 65–74.
    """

    def __init__(self, n_components=5, whiten=False, random_state=42):
        """Initialize MNF extractor with transformation parameters."""
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.pca = None

    def _extract(self, data: HSI, **inputs):
        """
        Extract MNF features from a hyperspectral image.

        The algorithm estimates noise from spatial differences, performs noise
        whitening, and then applies PCA to maximize signal-to-noise ratio.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:

            - features : ndarray of shape (H, W, n_components)
                MNF-transformed features.
            - explained_variance_ratio : ndarray of shape (n_components,)
                Percentage of variance explained by each component.
            - explained_variance : ndarray of shape (n_components,)
                Variance of each component.
            - components : ndarray of shape (n_components, n_features)
                PCA components after noise whitening.
            - mean : ndarray of shape (n_features,)
                Mean of each feature before whitening.
            - n_components : int
                Number of components used in the transformation.
            - original_shape : tuple of int
                Shape of the original hyperspectral cube (H, W, bands).
            - cumulative_variance_ratio : ndarray of shape (n_components,)
                Cumulative explained variance ratio.
            - noise_eigenvalues : ndarray of shape (n_features,)
                Eigenvalues of the estimated noise covariance.
            - whitening_matrix : ndarray of shape (n_features, n_features)
                Whitening matrix applied to the data.
        """
        X = data.reflectance
        h, w, bands = X.shape
        X_reshaped = X.reshape(-1, bands)

        # Center data
        X_mean = np.mean(X_reshaped, axis=0)
        X_centered = X_reshaped - X_mean

        # Noise estimation
        diffs_h = X[1:, :, :] - X[:-1, :, :]  # vertical differences
        diffs_w = X[:, 1:, :] - X[:, :-1, :]  # horizontal differences
        diffs = np.concatenate(
            [diffs_h.reshape(-1, bands), diffs_w.reshape(-1, bands)], axis=0
        )
        noise_cov = np.cov(diffs.T)

        # Whitening matrix
        eigvals, eigvecs = np.linalg.eigh(noise_cov)
        eigvals = np.clip(eigvals, 1e-12, None)
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Noise pre-whitening
        X_whitened = X_centered @ W.T

        # Apply PCA on whitened data
        actual_n_components = min(self.n_components, X_whitened.shape[1])
        self.pca = PCA(
            n_components=actual_n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )

        features_2d = self.pca.fit_transform(X_whitened)
        features = features_2d.reshape(h, w, actual_n_components)

        return {
            "features": features,
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
            "explained_variance": self.pca.explained_variance_,
            "components": self.pca.components_,
            "mean": X_mean,
            "n_components": self.pca.n_components_,
            "original_shape": (h, w, bands),
            "cumulative_variance_ratio": np.cumsum(
                self.pca.explained_variance_ratio_
            ),
            "noise_eigenvalues": eigvals,
            "whitening_matrix": W,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if not isinstance(self.whiten, bool):
            raise ValueError("whiten must be boolean")
