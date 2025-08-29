from abc import abstractmethod
from typing import Any, Dict
import numpy as np

from hyppo.hsi import HSI
from .base import Extractor


class CombinedExtractor(Extractor):
    """Base class for extractors that combine spectral and spatial information.
    
    Combined extractors compute features that require both spectral and spatial
    context, such as spectral indices computed over spatial neighborhoods.
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    def extract(self, data: HSI) -> Dict[str, Any]:
        # Extract combined spectral-spatial features
        features = self.extract_combined(data)
        
        return {
            "features": features,
            "shape": data.shape,
            "wavelengths": data.wavelengths
        }
    
    @abstractmethod
    def extract_combined(self, data: HSI) -> np.ndarray:
        """Extract features combining spectral and spatial information.
        
        Args:
            data: HSI object containing the hyperspectral image
            
        Returns:
            Extracted combined features
        """
        pass