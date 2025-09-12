from .base import Extractor
from hyppo.core import HSI
import numpy as np
import pywt
from skimage.transform import resize


class DWT3DExtractor(Extractor):
    """
    Discrete Wavelet Transform (3D) feature extractor for hyperspectral images.

    Applies 3D DWT along the spatial (H, W) and spectral (B) dimensions of the hyperspectral cube.
    Each level of decomposition produces an approximation (LLL) and detail coefficients (subbands)
    that are upsampled to the original resolution and concatenated along the feature axis.

    Parameters
    ----------
    wavelet : str, default='db4'
        Mother wavelet used for decomposition.
    mode : str, default='symmetric'
        Signal extension mode for wavelet transform.
    levels : int, default=1
        Number of decomposition levels.

    References
    ----------
    Qian, Ye, and Zhou (2012): Decomposed hyperspectral images at different scales,
        orientations, and frequencies using 3-D wavelets, and applied sparse logistic
        regression for feature selection and classification.
    Ye et al. (2014): Extracted 3-D DWT coefficients to acquire spectral–spatial
        information for classification, using subspace division to improve class separation
        and reduce training samples.

    """

    def __init__(self, wavelet="db4", mode="symmetric", levels=1):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.levels = levels

    def extract(self, data: HSI):
        """
        Extract 3D DWT features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:
                - "features": np.ndarray of shape (H, W, B * n_subbands)
                    Upsampled wavelet coefficient maps stacked along the last axis.
                - "wavelet": str, wavelet used for decomposition
                - "mode": str, signal extension mode
                - "levels": int, number of decomposition levels
                - "n_features": int, total number of features per pixel
                - "original_shape": tuple, original shape of the HSI cube (H, W, B)
        """
        X = data.reflectance()
        h, w, b = X.shape
        original_shape = X.shape

        # Apply 3D DWT
        coeffs = pywt.wavedecn(
            X, wavelet=self.wavelet, mode=self.mode, level=self.levels
        )

        # coeffs[0] is the LLL approximation
        # coeffs[1:] are dictionaries with subbands like 'aad', 'ada', ..., 'ddd'
        all_subbands = []

        # Upsample LLL
        cA = coeffs[0]
        cA_up = resize(cA, (h, w, b), order=1, mode="reflect", anti_aliasing=False)
        all_subbands.append(cA_up)

        # Process detail coefficients
        for detail_level in coeffs[1:]:
            for key, c in detail_level.items():
                c_up = resize(
                    c, (h, w, b), order=1, mode="reflect", anti_aliasing=False
                )
                all_subbands.append(c_up)

        # Stack all coefficients along the feature axis (last dimension)
        features = np.concatenate(
            all_subbands, axis=-1
        )  # shape: (h, w, b * n_subbands)

        return {
            "features": features,
            "wavelet": self.wavelet,
            "mode": self.mode,
            "levels": self.levels,
            "n_features": features.shape[-1],
            "original_shape": original_shape,
        }

    def validate(self):
        """Validate extractor parameters."""
        if self.wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{self.wavelet}' not available")
        if self.mode not in pywt.Modes.modes:
            raise ValueError(f"Mode '{self.mode}' not available")
        if not isinstance(self.levels, int) or self.levels <= 0:
            raise ValueError("levels must be a positive integer")
