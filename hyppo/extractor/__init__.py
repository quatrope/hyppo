"""Feature extractors for hyperspectral image processing."""

from .base import Extractor
from .dwt1d import DWT1DExtractor
from .dwt2d import DWT2DExtractor
from .dwt3d import DWT3DExtractor
from .gabor import GaborExtractor
from .geometricmoment import GeometricMomentExtractor
from .glcm import GLCMExtractor
from .ica import ICAExtractor
from .lbp import LBPExtractor
from .legendremoment import LegendreMomentExtractor
from .max import MaxExtractor
from .mean import MeanExtractor
from .median import MedianExtractor
from .min import MinExtractor
from .mnf import MNFExtractor
from .mp import MPExtractor
from .ndvi import NDVIExtractor
from .ndwi import NDWIExtractor
from .pca import PCAExtractor
from .pp import PPExtractor
from .registry import registry
from .savi import SAVIExtractor
from .std import StdExtractor
from .zernikemoment import ZernikeMomentExtractor

__all__ = [
    "Extractor",
    "registry",
    "DWT1DExtractor",
    "DWT2DExtractor",
    "DWT3DExtractor",
    "GaborExtractor",
    "GeometricMomentExtractor",
    "GLCMExtractor",
    "ICAExtractor",
    "LBPExtractor",
    "LegendreMomentExtractor",
    "MaxExtractor",
    "MeanExtractor",
    "MedianExtractor",
    "MinExtractor",
    "MNFExtractor",
    "MPExtractor",
    "NDVIExtractor",
    "NDWIExtractor",
    "PCAExtractor",
    "PPExtractor",
    "SAVIExtractor",
    "StdExtractor",
    "ZernikeMomentExtractor",
]

registry.register(DWT1DExtractor)
registry.register(DWT2DExtractor)
registry.register(DWT3DExtractor)
registry.register(GaborExtractor)
registry.register(GeometricMomentExtractor)
registry.register(GLCMExtractor)
registry.register(ICAExtractor)
registry.register(LBPExtractor)
registry.register(LegendreMomentExtractor)
registry.register(MaxExtractor)
registry.register(MeanExtractor)
registry.register(MedianExtractor)
registry.register(MinExtractor)
registry.register(MNFExtractor)
registry.register(MPExtractor)
registry.register(NDVIExtractor)
registry.register(NDWIExtractor)
registry.register(PCAExtractor)
registry.register(PPExtractor)
registry.register(SAVIExtractor)
registry.register(StdExtractor)
registry.register(ZernikeMomentExtractor)
