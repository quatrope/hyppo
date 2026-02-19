"""Pytest configuration HYPPO tests."""

import logging

from .fixtures.hsi import large_spectral_hsi, sample_hsi, sample_hsi_data, small_hsi

__all__ = ["large_spectral_hsi", "sample_hsi", "small_hsi", "sample_hsi_data"]

# Silence dask distributed logging noise during test teardown
logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("distributed.scheduler").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
logging.getLogger("distributed.core").setLevel(logging.CRITICAL)
