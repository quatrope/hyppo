from abc import abstractmethod
from typing import Any, Dict
import numpy as np

from hyppo.hsi import HSI
from .base import Extractor


class SpectralExtractor(Extractor):
    """Base class for extractors that operate on spectral signatures.
    
    Spectral extractors compute features from the spectral dimension of each pixel,
    treating each pixel as an independent spectral signature.
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    def extract(self, data: HSI) -> Dict[str, Any]:
        # Get the reflectance data and reshape for spectral processing
        reflectance = data.reflectance
        mask = data.mask
        
        # Reshape to (n_pixels, n_bands) for easier spectral processing
        n_pixels = data.height * data.width
        pixels_flat = reflectance.reshape(n_pixels, data.n_bands)
        mask_flat = mask.reshape(n_pixels)
        
        # Extract spectral features for valid pixels
        features = self.extract_spectral(pixels_flat, mask_flat, data.wavelengths)
        
        # Reshape back to spatial dimensions if needed
        if isinstance(features, np.ndarray) and features.shape[0] == n_pixels:
            features = features.reshape(data.height, data.width, -1)
        
        return {"features": features, "wavelengths": data.wavelengths}
    
    @abstractmethod
    def extract_spectral(self, pixels: np.ndarray, mask: np.ndarray, 
                        wavelengths: np.ndarray) -> np.ndarray:
        """Extract spectral features from pixel signatures.
        
        Args:
            pixels: Array of shape (n_pixels, n_bands) containing spectral signatures
            mask: Boolean array of shape (n_pixels,) indicating valid pixels
            wavelengths: Array of shape (n_bands,) with wavelength values
            
        Returns:
            Extracted features, typically of shape (n_pixels,) or (n_pixels, n_features)
        """
        pass