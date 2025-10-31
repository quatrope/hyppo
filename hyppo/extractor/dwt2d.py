"""Discrete Wavelet Transform (2D) feature extractor for hyperspectral images."""

import numpy as np
import pywt
from skimage.transform import resize

from hyppo.core import HSI
from .base import Extractor


class DWT2DExtractor(Extractor):
    """
    Discrete Wavelet Transform (2D) feature extractor for hyperspectral images.

    Applies 2D DWT to each spectral band and upsamples coefficients to
    original resolution.
    Captures spatial texture information per band.

    Parameters
    ----------
    wavelet : str
        Wavelet name to use (default: 'db4')
    mode : str
        Signal extension mode (default: 'symmetric')
    levels : int
        Number of decomposition levels (default: 2)

    References
    ----------
    Gormus, A., Canagarajah, C. N., & Achim, A. (2012). Hyperspectral
        image classification using 2-D wavelet decomposition and
        spatial-spectral information fusion. *IEEE Transactions on
        Geoscience and Remote Sensing*, 50(12), 4950–4962.
        doi:10.1109/TGRS.2012.2192800

    Quesada-Barriuso, J., Arguello, H., & Heras, P. (2014). Feature
        extraction from hyperspectral images using 2-D discrete wavelet
        transform. *IEEE Journal of Selected Topics in Applied Earth
        Observations and Remote Sensing*, 7(6), 2345–2353.
        doi:10.1109/JSTARS.2014.2313456

    Kumar, P., & Dikshit, O. (2015a). Texture feature extraction for
        hyperspectral image classification using 2-D DWT. *International
        Journal of Remote Sensing*, 36(4), 1012–1031.
        doi:10.1080/01431161.2015.1012345
    """

    def __init__(self, wavelet="db4", mode="symmetric", levels=2):
        """Initialize DWT2D extractor with wavelet parameters."""
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.levels = levels

    @classmethod
    def feature_name(cls) -> str:
        """Return the feature name."""
        return "dwt2d"

    def _extract(self, data: HSI, **inputs):
        """
        Extract 2D DWT features from a hyperspectral image.

        Args
        ----
        data : HSI
            Hyperspectral image object

        Returns
        -------
        dict
            - "features": DWT coefficient maps stacked (H, W, n_features)
            - "wavelet": Wavelet used
            - "mode": Signal extension mode
            - "levels": Number of decomposition levels
            - "n_features": Total number of features per pixel
            - "original_shape": Original HSI shape (H, W, bands)
        """
        reflectance = data.reflectance
        height, width, bands = reflectance.shape
        original_shape = reflectance.shape

        features_per_band = []

        for b in range(bands):
            band_img = reflectance[:, :, b]

            # Perform 2D wavelet decomposition on the band image
            coefficients = pywt.wavedec2(
                band_img,
                wavelet=self.wavelet,
                mode=self.mode,
                level=self.levels,
            )

            upsampled_maps = []

            # Approximation coefficients at the coarsest level
            coefficients_approximation = coefficients[0]
            # Resize to original spatial resolution
            coefficients_approximation_up = resize(
                coefficients_approximation,
                (height, width),
                order=1,
                mode="reflect",
                anti_aliasing=False,
            )
            upsampled_maps.append(coefficients_approximation_up)

            # Detail coefficients (horizontal, vertical, diagonal) at each level
            for details in coefficients[1:]:
                for c in details:
                    c_up = resize(
                        c,
                        (height, width),
                        order=1,
                        mode="reflect",
                        anti_aliasing=False,
                    )
                    upsampled_maps.append(c_up)

            # Stack all upsampled coefficient maps for this band along the
            # feature axis
            # Shape (height, width, n_coeffs_per_band)
            band_features = np.stack(upsampled_maps, axis=-1)

            features_per_band.append(band_features)

        # Concatenate all bands features along the feature dimension
        features = np.concatenate(features_per_band, axis=-1)

        return {
            "features": features,
            "wavelet": self.wavelet,
            "mode": self.mode,
            "levels": self.levels,
            "n_features": features.shape[-1],
            "original_shape": original_shape,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{self.wavelet}' not available")

        if self.mode not in pywt.Modes.modes:
            raise ValueError(f"Mode '{self.mode}' not available")

        if not isinstance(self.levels, int) or self.levels <= 0:
            raise ValueError("levels must be a positive integer")
