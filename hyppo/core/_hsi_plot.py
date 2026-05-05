"""Plotting accessor for HSI."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from hyppo.core._hsi import HSI


class HSIPlotAccessor:
    """Accessor exposing plotting helpers on an HSI instance."""

    def __init__(self, hsi: "HSI"):
        """Initialize with backreference to the parent HSI."""
        self._hsi = hsi

    def pseudo_rgb(
        self,
        ax=None,
        r_nm: float = 650.0,
        g_nm: float = 550.0,
        b_nm: float = 450.0,
        percentile_clip: tuple = (2, 98),
        title: str | None = None,
    ):
        """Plot pseudo-RGB composite of the closest R/G/B bands."""
        rgb_hsi = self._hsi.pseudo_rgb(r_nm=r_nm, g_nm=g_nm, b_nm=b_nm)
        rgb = rgb_hsi.reflectance.astype(np.float32)
        p_low, p_high = np.percentile(rgb, percentile_clip)
        rgb = np.clip((rgb - p_low) / (p_high - p_low + 1e-9), 0, 1)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)
        else:
            wls = rgb_hsi.wavelengths
            ax.set_title(
                f"Pseudo-RGB. Bands @ "
                f"{wls[0]:.0f}/{wls[1]:.0f}/{wls[2]:.0f} nm"
            )
        return ax
