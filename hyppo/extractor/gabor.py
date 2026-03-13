"""Gabor filter feature extractor for pixel-wise hyperspectral image classification."""

import numpy as np
from scipy import ndimage

from hyppo.core import HSI
from .base import Extractor
from ._validators import validate_positive_int


class GaborExtractor(Extractor):
    """
    Gabor filter feature extractor for pixel-wise hyperspectral classification.

    Applies a bank of Gabor filters (M scales x N orientations) over each
    selected spectral band and returns a spatial feature map (H, W, n_features).
    Each pixel is characterized by the filter responses at all scales and
    orientations, following Rajadell et al. (2013) Section II-B.

    Optionally computes opponent features (Rajadell Eq. 4) as the direct
    difference of filter responses between pairs of spectral bands.

    Parameters
    ----------
    n_scales : int, default=2
        Number of scales. Rajadell et al. use up to 6, but recommend 2-3
        for most hyperspectral land-cover datasets (Section III-D).
    n_orientations : int, default=4
        Number of orientations. 4 (0, 45, 90, 135 deg) is the minimum
        recommended for full texture coverage (Rajadell Section III-B).
    use_opponent : bool, default=False
        If True, also compute pairwise opponent features (Rajadell Eq. 4).
        NOTE: scales as O(B^2 * M * N). Only use after band selection (B<=10).
        For B=3, M=2, N=4 this adds 24 extra feature maps.
    sigmas_sq : list of float or None, default=None
        Gaussian envelope variances (sigma^2) for each scale.
        If None, uses the Shi & Healey (2003) values [1.97, 7.89] for
        n_scales<=2, and extends with dyadic values [4^m] for more scales.

        These control the spatial support (kernel size) of each filter:
          - Small sigma_sq (e.g. 1.97, kernel ~9px): sensitive to fine detail
            and edges; spectral peak near 0.5 cycles/pixel (high frequency).
          - Large sigma_sq (e.g. 7.89, kernel ~23px): sensitive to coarse
            texture; spectral peak near 0.25 cycles/pixel (medium frequency).

        For AVIRIS-style land-cover images (20m/pixel, homogeneous regions),
        consider larger values like [4, 16] to target lower spatial frequencies.
    frequencies : list of float or None, default=None
        Central spatial frequency (cycles/pixel) for each scale.
        If None, uses dyadic values: u_m = 0.5 / 2^m => [0.5, 0.25, 0.125, ...]
        Must have the same length as sigmas_sq if both are provided.

        Rule of thumb: the filter responds well to textures whose spatial
        frequency matches u_m +/- (u_m / 2).

    Notes
    -----
    Band selection MUST be applied upstream before calling this extractor
    when use_opponent=True. With 126 bands the opponent features would
    generate ~63,000 maps, which is computationally prohibitive.

    The default parameters (sigmas_sq from Shi & Healey, dyadic frequencies)
    work well as a starting point but should be validated against the actual
    spatial frequency content of the target image.

    References
    ----------
    Rajadell, O., Garcia-Sevilla, P., & Pla, F. (2013). Spectral-Spatial
    Pixel Characterization Using Gabor Filters for Hyperspectral Image
    Classification. IEEE Geoscience and Remote Sensing Letters, 10(4), 860-864.

    Shi, M., & Healey, G. (2003). Hyperspectral Texture Recognition Using a
    Multiscale Opponent Representation. IEEE Transactions on Geoscience and
    Remote Sensing, 41(5), 1090-1095.
    """

    # Default sigma values from Shi & Healey (2003), Section II.
    # sigma_sq = (1.97, 7.89) -> sigma ~ (1.40, 2.81)
    _DEFAULT_SIGMAS_SQ = [1.97, 7.89]

    def __init__(
        self,
        n_scales=2,
        n_orientations=4,
        use_opponent=False,
        sigmas_sq=None,
        frequencies=None,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.use_opponent = use_opponent
        self.sigmas_sq = sigmas_sq
        self.frequencies = frequencies

    @classmethod
    def feature_name(cls):
        return "gabor"

    def _resolve_sigmas_sq(self):
        """Return sigma_sq list, resolving defaults if needed."""
        if self.sigmas_sq is not None:
            if len(self.sigmas_sq) != self.n_scales:
                raise ValueError(
                    f"sigmas_sq has {len(self.sigmas_sq)} values but "
                    f"n_scales={self.n_scales}. They must match."
                )
            return list(self.sigmas_sq)

        # Default: Shi & Healey values for first two scales, dyadic beyond
        defaults = self._DEFAULT_SIGMAS_SQ
        if self.n_scales <= len(defaults):
            return defaults[: self.n_scales]
        return defaults + [4.0**m for m in range(len(defaults), self.n_scales)]

    def _resolve_frequencies(self):
        """Return frequency list, resolving defaults if needed."""
        if self.frequencies is not None:
            if len(self.frequencies) != self.n_scales:
                raise ValueError(
                    f"frequencies has {len(self.frequencies)} values but "
                    f"n_scales={self.n_scales}. They must match."
                )
            return list(self.frequencies)

        # Default: dyadic u_m = 0.5 / 2^m
        return [0.5 / (2**m) for m in range(self.n_scales)]

    def _build_filter_bank(self):
        """
        Build Gabor kernels for all (scale, orientation) pairs.

        Returns
        -------
        kernels : list of (n_scales * n_orientations) 2D arrays
        param_list : list of (sigma_sq, freq, theta) tuples, same order
        """
        sigmas_sq = self._resolve_sigmas_sq()
        frequencies = self._resolve_frequencies()
        thetas = np.linspace(0, np.pi, self.n_orientations, endpoint=False)

        kernels, param_list = [], []
        for sigma_sq, freq in zip(sigmas_sq, frequencies):
            for theta in thetas:
                k = self._make_kernel(sigma_sq, freq, theta)
                kernels.append(k)
                param_list.append((sigma_sq, freq, theta))

        return kernels, param_list

    def _make_kernel(self, sigma_sq, freq, theta):
        """
        Build a single real Gabor kernel (Rajadell Eq. 2 / Shi & Healey Eq. 1).

        f_mn(x,y) = (1 / 2*pi*sigma^2)
                    * exp(-(x'^2 + y'^2) / (2*sigma^2))
                    * cos(2*pi*u_m * x')

        where x', y' are the rotated coordinates.
        """
        sigma = np.sqrt(sigma_sq)
        half = int(np.ceil(4 * sigma))  # covers >99% of Gaussian mass
        size = 2 * half + 1  # always odd

        coords = np.arange(-half, half + 1)
        y, x = np.meshgrid(coords, coords)

        x_rot = x * np.cos(theta) + y * np.sin(theta)
        y_rot = -x * np.sin(theta) + y * np.cos(theta)

        envelope = (1.0 / (2.0 * np.pi * sigma_sq)) * np.exp(
            -(x_rot**2 + y_rot**2) / (2.0 * sigma_sq)
        )
        carrier = np.cos(2.0 * np.pi * freq * x_rot)

        return (
            envelope * carrier
        )  # not normalised: preserves response magnitude

    def _extract(self, data: HSI, **inputs):
        """
        Extract Gabor feature maps from a hyperspectral image.

        For each spectral band and each filter in the bank, the full
        convolution response image is computed. The responses are stacked
        along the last axis to form the feature map (Rajadell Eq. 5).

        If use_opponent=True, pairwise difference maps are also appended
        (Rajadell Eq. 4): d^ij_mn = h^i_mn - h^j_mn

        Parameters
        ----------
        data : HSI
            Hyperspectral image. data.reflectance shape: (H, W, B).

        Returns
        -------
        dict with keys:
            "features"     : ndarray (H, W, n_features)
            "n_unichrome"  : int
            "n_opponent"   : int  (0 if use_opponent=False)
            "n_features"   : int
            "scales"       : int
            "orientations" : int
        """
        reflectance = data.reflectance
        H, W, B = reflectance.shape

        kernels, _ = self._build_filter_bank()
        n_filters = len(kernels)  # n_scales * n_orientations

        # Unichrome: h^i_mn = I_i * f_mn  (Rajadell Eq. 5 / Shi Eq. 2)
        responses = np.empty((B, n_filters, H, W), dtype=np.float32)
        for i in range(B):
            band = reflectance[:, :, i].astype(np.float32)
            for k, kernel in enumerate(kernels):
                responses[i, k] = ndimage.convolve(
                    band, kernel.astype(np.float32), mode="reflect"
                )

        # Stack: (H, W, B * n_filters)
        # Axis order: all filters for band 0, then band 1, ...
        unichrome_maps = responses.transpose(2, 3, 0, 1).reshape(
            H, W, B * n_filters
        )

        features_list = [unichrome_maps]
        n_opponent = 0

        # Opponent: d^ij_mn = h^i_mn - h^j_mn  (Rajadell Eq. 4)
        if self.use_opponent:
            opponent_maps = []
            for i in range(B):
                for j in range(i + 1, B):
                    for k in range(n_filters):
                        opponent_maps.append(responses[i, k] - responses[j, k])
            if opponent_maps:
                opp_stack = np.stack(opponent_maps, axis=-1)
                features_list.append(opp_stack)
                n_opponent = opp_stack.shape[-1]

        features = np.concatenate(features_list, axis=-1)

        return {
            "features": features,
            "n_unichrome": B * n_filters,
            "n_opponent": n_opponent,
            "n_features": features.shape[-1],
            "scales": self.n_scales,
            "orientations": self.n_orientations,
            "original_shape": reflectance.shape,
        }

    def _validate(self, data: HSI, **inputs):
        validate_positive_int(self.n_scales, "n_scales")
        validate_positive_int(self.n_orientations, "n_orientations")

        # Eagerly resolve to catch length mismatches before extraction
        self._resolve_sigmas_sq()
        self._resolve_frequencies()

        if self.use_opponent:
            _, _, B = data.reflectance.shape
            if B > 10:
                raise ValueError(
                    f"use_opponent=True with {B} bands would generate "
                    f"{B*(B-1)//2 * self.n_scales * self.n_orientations} opponent maps. "
                    "Apply band selection upstream to reduce B <= 10 before "
                    "using opponent features."
                )
