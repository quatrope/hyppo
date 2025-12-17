"""Principal Component Analysis (PCA) feature extractor.

For hyperspectral images.
"""

import numpy as np
from sklearn.decomposition import PCA

from hyppo.core import HSI
from .base import Extractor


class PCAExtractor(Extractor):
    """Principal Component Analysis (PCA) feature extractor.

    Applies PCA transformation [1]_ to reduce dimensionality while preserving
    maximum variance in the spectral data.

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
    .. [1] Jolliffe, I.T., "Principal Component Analysis", Springer, 2002.
    """

    def __init__(self, n_components=5, whiten=False, random_state=42):
        """Initialize PCA extractor with decomposition parameters."""
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.pca = None
    
    @classmethod
    def feature_name(cls) -> str:
        """Return the feature name."""
        return "pca"

    def _extract(self, data: HSI, **inputs):
        """
        Extract PCA features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:

            - features : ndarray of shape (H, W, n_components)
                PCA-transformed features.
            - explained_variance_ratio : ndarray of shape (n_components,)
                Percentage of variance explained by each component.
            - explained_variance : ndarray of shape (n_components,)
                Variance of each principal component.
            - components : ndarray of shape (n_components, n_features)
                Principal axes in feature space.
            - mean : ndarray of shape (n_features,)
                Per-feature empirical mean.
            - n_components : int
                Number of components used in the transformation.
            - original_shape : tuple of int
                Shape of the original hyperspectral cube (H, W, bands).
            - cumulative_variance_ratio : ndarray of shape (n_components,)
                Cumulative explained variance ratio.
        """
        X = data.reflectance

        # Prepare data
        h, w, bands = X.shape  # (height, width, bands) -> (pixels, bands)
        X_reshaped = X.reshape(-1, bands)

        # Verify that n_components is valid
        n_samples, n_features = X_reshaped.shape
        max_components = min(n_samples, n_features)
        actual_n_components = min(self.n_components, max_components)

        # PCA
        self.pca = PCA(
            n_components=actual_n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )

        # Transform data
        features_2d = self.pca.fit_transform(X_reshaped)
        features = features_2d.reshape(h, w, actual_n_components)

        return {
            "features": features,
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
            "explained_variance": self.pca.explained_variance_,
            "components": self.pca.components_,
            "mean": self.pca.mean_,
            "n_components": actual_n_components,
            "original_shape": (h, w, bands),
            "cumulative_variance_ratio": np.cumsum(
                self.pca.explained_variance_ratio_
            ),
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if not isinstance(self.whiten, bool):
            raise ValueError("whiten must be boolean")
