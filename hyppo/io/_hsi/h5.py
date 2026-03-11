from pathlib import Path

import h5py
import numpy as np

from hyppo.core import HSI


def load_h5_hsi(
    path: Path | str,
    reflectance_path: str | None = None,
    wavelength_path: str | None = None,
) -> HSI:
    """Load hyperspectral image from H5 file."""
    if not isinstance(path, Path):
        path = Path(path)

    file_format = path.suffix
    if file_format != ".h5":
        raise ValueError(f"Unknown Hyper Spectral Image format: {file_format}")

    with h5py.File(path, "r") as f:
        parsed_data = _parse_h5_hsi(f, reflectance_path, wavelength_path)

        reflectance = parsed_data["reflectance_dataset"][:]
        wavelengths = parsed_data["wavelength_dataset"][:]
        scale_factor = parsed_data["scale_factor"]

        scaled_reflectance = reflectance.astype(np.float32) / scale_factor

        if "null_value" in parsed_data:
            mask = np.all(
                scaled_reflectance != parsed_data["null_value"], axis=2
            )
        else:
            mask = np.ones(scaled_reflectance.shape[:2], dtype=bool)

        metadata = {
            "file_path": str(path),
            "reflectance_name": parsed_data["reflectance_name"],
            "wavelength_name": parsed_data["wavelength_name"],
            "scale_factor": scale_factor,
        }

        return HSI(
            reflectance=scaled_reflectance,
            wavelengths=wavelengths,
            mask=mask,
            metadata=metadata,
        )


def _extract_dataset_attrs(ref_dataset, scale_factor_name, null_value_name):
    """Extract scale factor, null value, and metadata from dataset attrs."""
    result = {}

    scale_factor = ref_dataset.attrs.get(scale_factor_name)
    if scale_factor:
        result["scale_factor"] = scale_factor[0]

    null_value = ref_dataset.attrs.get(null_value_name)
    if null_value:
        result["null_value"] = null_value[0]

    metadata = {}
    for key, value in ref_dataset.attrs.items():
        metadata[key] = value
    result["reflectance_metadata"] = metadata

    return result


def _parse_h5_hsi(
    file: h5py.File,
    reflectance_name: str | None = None,
    wavelength_name: str | None = None,
    scale_factor_name: str | None = None,
    null_value_name: str | None = None,
):
    if scale_factor_name is None:
        scale_factor_name = "Scale_Factor"
    if null_value_name is None:
        null_value_name = "Data_Ignore_Value"

    ref_dataset, ref_path = _find_reflectance_dataset(file, reflectance_name)
    wave_dataset, wave_path = _find_wavelength_dataset(file, wavelength_name)

    if ref_dataset is None:
        raise ValueError("Could not find reflectance dataset in H5 file")

    if wave_dataset is None:
        raise ValueError("Could not find wavelength dataset in H5 file")

    parsed_data = {
        "reflectance_dataset": ref_dataset,
        "reflectance_name": ref_path,
        "wavelength_dataset": wave_dataset,
        "wavelength_name": wave_path,
        "scale_factor": 1.0,
        "reflectance_metadata": {},
    }

    attrs = _extract_dataset_attrs(
        ref_dataset, scale_factor_name, null_value_name
    )
    parsed_data.update(attrs)

    return parsed_data


def _validate_provided_dataset(f, provided_path, ndim, label):
    """Validate a user-provided dataset path exists and has correct ndim."""
    if provided_path not in f:
        raise ValueError(f"Provided {label} path '{provided_path}' is invalid")

    node = f[provided_path]
    if not isinstance(node, h5py.Dataset) or node.ndim != ndim:
        raise ValueError(f"Provided {label} path '{provided_path}' is invalid")

    return node


def _reflectance_priority(name_lower):
    """Return priority for reflectance candidate (1=best, 2=fallback)."""
    if "reflectance" in name_lower or "reflectancia" in name_lower:
        return 1
    return 2


def _find_reflectance_dataset(
    f: h5py.File, provided_path: str | None = None
) -> tuple[h5py.Dataset | None, str]:

    if provided_path:
        node = _validate_provided_dataset(f, provided_path, 3, "reflectance")
        return node, provided_path

    candidates = []

    def visitor(name, node):
        if isinstance(node, h5py.Dataset) and node.ndim == 3:
            name_lower = name.lower()
            keywords = ["reflectance", "reflectancia", "radiance"]
            if any(kw in name_lower for kw in keywords):
                priority = _reflectance_priority(name_lower)
                candidates.append((priority, name, node))

    f.visititems(visitor)

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][2], candidates[0][1]

    return None, ""


def _find_wavelength_dataset(
    f: h5py.File, provided_path: str | None = None
) -> tuple[h5py.Dataset | None, str]:

    if provided_path:
        node = _validate_provided_dataset(f, provided_path, 1, "wavelength")
        return node, provided_path

    candidates = []

    def visitor(name, node):
        if isinstance(node, h5py.Dataset) and node.ndim == 1:
            if "wavelength" in name.lower():
                candidates.append((name, node))

    f.visititems(visitor)

    if candidates:
        return candidates[0][1], candidates[0][0]

    return None, ""
