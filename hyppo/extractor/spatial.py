from abc import abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from hyppo.hsi import HSI
from .base import Extractor


class SpatialExtractor(Extractor):
    """Base class for extractors that operate on spatial patterns.

    Spatial extractors compute features from the spatial dimensions of the image,
    typically working on individual bands or band combinations.
    """

    def __init__(self, band_indices: Optional[list] = None) -> None:
        super().__init__()
        self.band_indices = band_indices

    def extract(self, data: HSI) -> Dict[str, Any]:
        # Determine which bands to process
        if self.band_indices is None:
            # Process all bands by default
            band_indices = list(range(data.n_bands))
        else:
            band_indices = self.band_indices

        # Validate band indices
        for idx in band_indices:
            if not 0 <= idx < data.n_bands:
                raise ValueError(f"Band index {idx} out of range [0, {data.n_bands})")

        # Extract spatial features for selected bands
        features = self.extract_spatial(data, band_indices)

        return {
            "features": features,
            "band_indices": band_indices,
            "wavelengths": (
                data.wavelengths[band_indices] if len(band_indices) > 0 else []
            ),
        }

    @abstractmethod
    def extract_spatial(self, data: HSI, band_indices: list) -> np.ndarray:
        """Extract spatial features from selected bands.

        Args:
            data: HSI object containing the hyperspectral image
            band_indices: List of band indices to process

        Returns:
            Extracted spatial features
        """
        pass
