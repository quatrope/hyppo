"""HYPPO: Hyperspectral Processing for feature extraction."""

import importlib.metadata

import hyppo.core as core
import hyppo.extractor as extractor
import hyppo.io as io
import hyppo.runner as runner

NAME = "hyppo-hsi"

__version__ = importlib.metadata.version(NAME)

__all__ = ["core", "runner", "io", "extractor", "__version__"]

del importlib
