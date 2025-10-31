"""Independent Component Analysis (ICA) feature extractor for hyperspectral images."""

import numpy as np
from sklearn.decomposition import FastICA

from hyppo.core import HSI
from .base import Extractor


class ICAExtractor(Extractor):
    """
    Independent Component Analysis (ICA) feature extractor for hyperspectral
    images.

    Parameters
    ----------
    n_components : int, default=5
        Number of components to keep.
    whiten : {'unit-variance', 'arbitrary-variance', False}, default='unit-variance'
        Whitening strategy to apply before ICA.
    random_state : int, RandomState instance or None, default=42
        Random state for reproducibility.

    References
    ----------
    .. [1] Hyvärinen, A., & Oja, E. (2000). Independent component analysis:
       algorithms and applications. Neural Networks, 13(4–5), 411–430.
    """

    def __init__(
        self, n_components=5, whiten="unit-variance", random_state=42
    ):
        """Initialize ICA extractor with decomposition parameters."""
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.ica = None

    def _extract(self, data: HSI, **inputs):
        """
        Extract ICA features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:

            - features : ndarray of shape (H, W, n_components)
                ICA-transformed features.
            - components : ndarray of shape (n_components, n_features)
                Independent components found by ICA.
            - mixing_matrix : ndarray of shape (n_features, n_components)
                Estimated mixing matrix.
            - mean : ndarray of shape (n_features,) or None
                Per-feature mean if available.
            - n_components : int
                Number of components used in the transformation.
            - original_shape : tuple of int
                Shape of the original hyperspectral cube (H, W, bands).
            - n_iter : int
                Number of iterations run by ICA.
            - reconstruction_error : float or None
                Mean squared reconstruction error (if computable).
        """
        reflectance = data.reflectance
        height, width, bands = reflectance.shape
        reflectance_reshaped = reflectance.reshape(-1, bands)

        # Verify that n_components is valid
        n_samples, n_features = reflectance_reshaped.shape
        max_components = min(n_samples, n_features)
        actual_n_components = min(self.n_components, max_components)

        # ICA
        self.ica = FastICA(
            n_components=actual_n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )

        # Transform data
        features_2d = self.ica.fit_transform(reflectance_reshaped)

        features = features_2d.reshape(height, width, actual_n_components)

        # Calculate mixing matrix
        mixing_matrix = self.ica.mixing_

        # Calculate reconstruction error if possible
        try:
            X_reconstructed = self.ica.inverse_transform(features_2d)
            reconstruction_error = np.mean(
                (reflectance_reshaped - X_reconstructed) ** 2
            )
        except Exception:  # TODO! Specifiy proper exception
            reconstruction_error = None

        return {
            "features": features,
            "components": self.ica.components_,
            "mixing_matrix": mixing_matrix,
            "mean": self.ica.mean_ if hasattr(self.ica, "mean_") else None,
            "n_components": actual_n_components,
            "original_shape": (height, width, bands),
            "n_iter": self.ica.n_iter_,
            "reconstruction_error": reconstruction_error,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")

        valid_whiten = ["unit-variance", "arbitrary-variance", False]
        if self.whiten not in valid_whiten:
            raise ValueError(f"whiten must be one of {valid_whiten}")
