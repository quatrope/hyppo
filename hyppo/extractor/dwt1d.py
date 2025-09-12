from .base import Extractor
from hyppo.core import HSI
import numpy as np
import pywt


class DWT1DExtractor(Extractor):
    """
    Discrete Wavelet Transform (1D) feature extractor for hyperspectral images.

    Applies a 1D DWT to each pixel's spectral signature to extract
    multiscale spectral features.

    Parameters
    ----------
    wavelet : str, optional
        Wavelet name to use (default: 'db4').
    mode : str, optional
        Signal extension mode (default: 'symmetric').
    levels : int, optional
        Number of decomposition levels (default: 3).

    References
    ----------
    Bruce, K., Koger, C., & Li, J. (2002). Dimensionality reduction of hyperspectral data using discrete wavelet transform feature extraction. *IEEE Transactions on Geoscience and Remote Sensing*, 40(10), 2331–2338. https://doi.org/10.1109/TGRS.2002.804721

    Mallat, S. (1999). A Wavelet Tour of Signal Processing. Academic Press.
    """

    def __init__(self, wavelet="db4", mode="symmetric", levels=3):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.levels = levels

    def extract(self, data: HSI):
        """
        Extract 1D DWT features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:
                - "features" : ndarray
                    DWT-transformed array with shape (H, W, n_features).
                - "wavelet" : str
                    Wavelet used.
                - "mode" : str
                    Signal extension mode.
                - "levels" : int
                    Number of decomposition levels.
                - "coeffs_lengths" : list of int
                    Length of coefficients at each decomposition level.
                - "n_features" : int
                    Total number of features per pixel.
                - "original_shape" : tuple
                    Shape of the original HSI (H, W, bands).
        """
        # Prepare data
        X = data.reflectance()
        h, w, bands = X.shape
        X_reshaped = X.reshape(-1, bands)

        # Apply DWT to each pixel's spectral signature
        features_list = []

        for i in range(X_reshaped.shape[0]):
            pixel_spectrum = X_reshaped[i, :]

            # Apply 1D DWT to the spectral signature
            coeffs = pywt.wavedec(
                pixel_spectrum, self.wavelet, mode=self.mode, level=self.levels
            )

            # Concatenate all coefficients as features
            pixel_features = np.concatenate(coeffs)
            features_list.append(pixel_features)

        features_2d = np.array(features_list)
        features = features_2d.reshape(h, w, -1)

        # Get coefficient lengths from first pixel for reference
        sample_coeffs = pywt.wavedec(
            X_reshaped[0, :], self.wavelet, mode=self.mode, level=self.levels
        )
        coeffs_lengths = [len(c) for c in sample_coeffs]

        # ver si poner una validacion o algo con esto
        # max_level = pywt.dwt_max_level(data_len=107, filter_len=pywt.Wavelet('db4').dec_len)

        return {
            "features": features,
            "wavelet": self.wavelet,
            "mode": self.mode,
            "levels": self.levels,
            "coeffs_lengths": coeffs_lengths,
            "n_features": features.shape[1],
            "original_shape": (h, w, bands),
        }

    def validate(self):
        """Validate extractor parameters."""
        if self.wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{self.wavelet}' not available")
        if self.mode not in pywt.Modes.modes:
            raise ValueError(f"Mode '{self.mode}' not available")
        if not isinstance(self.levels, int) or self.levels <= 0:
            raise ValueError("levels must be a positive integer")


# La elección de la wavelet afecta la sensibilidad y la forma de la descomposición:
# - 'haar': detecta muy bien cambios abruptos y saltos en el espectro (ideal para bordes nítidos).
# - 'db4': captura variaciones suaves entre bandas de absorción, balanceando resolución y suavidad.
# - 'sym5': ofrece un buen balance entre suavidad y simetría, minimizando artefactos en los bordes.
# - 'coif2': modela formas suaves y complejas, pero genera más coeficientes y es más costosa computacionalmente.
#
# Además, el nivel de descomposición determina la profundidad con la que se analiza la señal:
# el máximo nivel posible depende de la longitud de la señal y del filtro wavelet,
# y puede obtenerse con pywt.dwt_max_level().
# En nuestro caso, con señal de longitud 107 y wavelet 'db4' (filtro de longitud 8),
# el nivel máximo permitido es 3, que es justamente el que usamos.
