"""Spectral band lookup utilities for vegetation index extractors."""

import warnings

import numpy as np


def find_band_index(wavelengths, target):
    """Return the index of the band closest to target wavelength."""
    return np.argmin(np.abs(wavelengths - target))


def warn_wavelength_tolerance(diffs_and_names, tolerance=50):
    """Emit a warning if any wavelength diff exceeds tolerance."""
    far_bands = [
        name for diff, name in diffs_and_names if diff > tolerance
    ]
    if far_bands:
        warnings.warn(
            "Bands far from target wavelengths: "
            + " ".join(far_bands)
        )


def find_and_validate_bands(data, band_targets):
    """Find closest bands and warn if far from targets.

    Parameters
    ----------
    data : HSI
        Hyperspectral image object.
    band_targets : list of (target_wavelength, band_name)
        Pairs of target wavelength and human-readable name.

    Returns
    -------
    list of (band_idx, band_array)
    """
    wavelengths = data.wavelengths
    if len(wavelengths) == 0:
        raise ValueError("No wavelength information available")

    results = []
    diffs_and_names = []
    for target_wl, name in band_targets:
        idx = find_band_index(wavelengths, target_wl)
        band = data.reflectance[:, :, idx].astype(float)
        diff = abs(wavelengths[idx] - target_wl)
        diffs_and_names.append((diff, name))
        results.append((idx, band))

    warn_wavelength_tolerance(diffs_and_names)
    return results
