#!/usr/bin/env python3
"""
Convert .mat hyperspectral datasets to .h5 format compatible with hyppo.

Usage:
    python convert_mat_to_h5.py
"""

from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"

# Dataset configurations
DATASETS = {
    "indian_pines": {
        "mat_file": "Indian_pines_corrected.mat",
        "gt_file": "Indian_pines_gt.mat",
        "data_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "wavelengths": np.linspace(400, 2500, 200),  # Approximate AVIRIS range
    },
    "pavia_university": {
        "mat_file": "PaviaU.mat",
        "gt_file": "PaviaU_gt.mat",
        "data_key": "paviaU",
        "gt_key": "paviaU_gt",
        "wavelengths": np.linspace(430, 860, 103),  # ROSIS sensor range
    },
    "houston_2013": {
        "mat_file": "Houston.mat",
        "gt_file": "Houston_gt.mat",
        "data_key": "Houston",
        "gt_key": "Houston_gt",
        "wavelengths": np.linspace(380, 1050, 144),  # ITRES CASI range
    },
}


def convert_dataset(name: str, config: dict) -> bool:
    """Convert a single dataset from .mat to .h5 format."""
    dataset_dir = DATA_DIR / name
    mat_path = dataset_dir / config["mat_file"]
    gt_path = dataset_dir / config["gt_file"]
    h5_path = dataset_dir / f"{name}.h5"

    print(f"\n{'='*50}")
    print(f"Converting: {name}")
    print(f"{'='*50}")

    if not mat_path.exists():
        print(f"  [SKIP] Data file not found: {mat_path}")
        return False

    # Load .mat file
    print(f"  Loading {mat_path.name}...")
    mat_data = loadmat(str(mat_path))

    # Find the data key (sometimes varies)
    data_key = config["data_key"]
    if data_key not in mat_data:
        # Try to find it
        possible_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        print(f"  Available keys: {possible_keys}")
        if len(possible_keys) == 1:
            data_key = possible_keys[0]
        else:
            print(f"  [ERROR] Cannot find data key '{config['data_key']}'")
            return False

    reflectance = mat_data[data_key].astype(np.float32)
    print(f"  Reflectance shape: {reflectance.shape}")

    # Load ground truth if available
    gt = None
    if gt_path.exists():
        print(f"  Loading {gt_path.name}...")
        gt_data = loadmat(str(gt_path))
        gt_key = config["gt_key"]
        if gt_key not in gt_data:
            possible_keys = [k for k in gt_data.keys() if not k.startswith("__")]
            if len(possible_keys) == 1:
                gt_key = possible_keys[0]
        if gt_key in gt_data:
            gt = gt_data[gt_key].astype(np.int32)
            print(f"  Ground truth shape: {gt.shape}")

    # Generate wavelengths
    n_bands = reflectance.shape[2]
    wavelengths = config["wavelengths"]
    if len(wavelengths) != n_bands:
        wavelengths = np.linspace(wavelengths[0], wavelengths[-1], n_bands)
    wavelengths = wavelengths.astype(np.float32)

    # Create mask (valid pixels where reflectance is not all zeros)
    mask = ~np.all(reflectance == 0, axis=2)

    # Normalize reflectance to [0, 1] range if needed
    if reflectance.max() > 1.0:
        scale_factor = 10000.0  # Common scale for hyperspectral data
        reflectance = reflectance / scale_factor
        print(f"  Normalized reflectance (scale factor: {scale_factor})")

    # Write H5 file
    print(f"  Writing {h5_path.name}...")
    with h5py.File(h5_path, "w") as f:
        # Main datasets
        f.create_dataset("reflectance", data=reflectance, compression="gzip")
        f.create_dataset("wavelengths", data=wavelengths)
        f.create_dataset("mask", data=mask)

        if gt is not None:
            f.create_dataset("ground_truth", data=gt)

        # Metadata
        f.attrs["name"] = name
        f.attrs["height"] = reflectance.shape[0]
        f.attrs["width"] = reflectance.shape[1]
        f.attrs["bands"] = reflectance.shape[2]
        f.attrs["wavelength_unit"] = "nm"

    # Verify
    file_size_mb = h5_path.stat().st_size / (1024 * 1024)
    print(f"  [OK] Created: {h5_path} ({file_size_mb:.1f} MB)")

    return True


def main():
    """Convert all available datasets."""
    print("=" * 50)
    print("MAT to H5 Converter for Hyperspectral Datasets")
    print("=" * 50)

    results = {}
    for name, config in DATASETS.items():
        results[name] = convert_dataset(name, config)

    # Summary
    print("\n" + "=" * 50)
    print("CONVERSION SUMMARY")
    print("=" * 50)
    for name, success in results.items():
        status = "✓ OK" if success else "✗ SKIP"
        print(f"  {name}: {status}")

    # List converted files
    print("\nConverted files:")
    for h5_file in DATA_DIR.glob("*/*.h5"):
        size_mb = h5_file.stat().st_size / (1024 * 1024)
        print(f"  {h5_file.relative_to(BENCHMARK_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
