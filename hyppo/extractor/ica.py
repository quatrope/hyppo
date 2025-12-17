"""Independent Component Analysis (ICA) feature extractor for HSI."""

import numpy as np
import warnings
from sklearn.decomposition import FastICA

from hyppo.core import HSI
from .base import Extractor


class ICAExtractor(Extractor):
    """
    Independent Component Analysis (ICA) extractor for hyperspectral images.

    Parameters
    ----------
    n_components : int, default=5
        Number of components to keep.
    whiten : {'unit-variance', 'arbitrary-variance', False}
        default='unit-variance'. Whitening strategy to apply before ICA.
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
                Unmixing matrix (ICA separating vectors).
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
        X = data.reflectance
        h, w, bands = X.shape
        X_flat = X.reshape(-1, bands)

        # Mean removal
        mean = X_flat.mean(axis=0)
        X_centered = X_flat - mean

        # Valid pixels
        valid_mask = np.isfinite(X_centered).all(axis=1)
        Xc_valid = X_centered[valid_mask]

        if self.whiten is False:
            warnings.warn(
                "ICA without whitening is generally ill-posed. "
                "Results may be unstable."
            )

        # Verify that n_components is valid
        n_samples, n_features = Xc_valid.shape
        n_comp = min(self.n_components, n_samples, n_features)

        # ICA
        self.ica = FastICA(
            n_components=n_comp,
            whiten=self.whiten,
            random_state=self.random_state,
        )

        # Transform data
        S = self.ica.fit_transform(Xc_valid)

        # Rebuild full feature map
        full_S = np.zeros((X_flat.shape[0], n_comp))
        full_S[valid_mask] = S
        features = full_S.reshape(h, w, n_comp)

        # Calculate reconstruction error
        try:
            X_reconstructed = self.ica.inverse_transform(S) + mean
            reconstruction_error = np.mean(
                (Xc_valid - X_reconstructed + mean) ** 2
            )
        except ValueError:
            reconstruction_error = None

        return {
            "features": features,
            "components": self.ica.components_,
            "mixing_matrix": self.ica.mixing_,
            "mean": mean,
            "n_components": n_comp,
            "original_shape": (h, w, bands),
            "n_iter": self.ica.n_iter_,
            "reconstruction_error": reconstruction_error,
            "valid_pixel_mask": valid_mask.reshape(h, w),
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")

        valid_whiten = ["unit-variance", "arbitrary-variance", False]
        if self.whiten not in valid_whiten:
            raise ValueError(f"whiten must be one of {valid_whiten}")
