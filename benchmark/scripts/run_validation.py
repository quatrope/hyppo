#!/usr/bin/env python3
"""
Validation script for hyppo SLURM benchmark.

Runs all extractors on each dataset with 2 SLURM jobs (8 cores each)
to verify the setup works before running the full benchmark.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import yaml

from hyppo import io
from hyppo.core import FeatureSpace
from hyppo.extractor import (
    DWT1DExtractor,
    DWT2DExtractor,
    DWT3DExtractor,
    GaborExtractor,
    GeometricMomentExtractor,
    GLCMExtractor,
    ICAExtractor,
    LBPExtractor,
    LegendreMomentExtractor,
    MNFExtractor,
    MPExtractor,
    NDVIExtractor,
    NDWIExtractor,
    PCAExtractor,
    PPExtractor,
    SAVIExtractor,
    ZernikeMomentExtractor,
)
from hyppo.runner import DaskSLURMRunner


BENCHMARK_DIR = Path(__file__).parent.parent
CONFIG_DIR = BENCHMARK_DIR / "config"
DATA_DIR = BENCHMARK_DIR / "data"
RESULTS_DIR = BENCHMARK_DIR / "results" / "validation"


def load_config():
    """Load benchmark configuration files."""
    with open(CONFIG_DIR / "datasets.yaml") as f:
        datasets_config = yaml.safe_load(f)

    with open(CONFIG_DIR / "slurm.yaml") as f:
        slurm_config = yaml.safe_load(f)

    return datasets_config, slurm_config


def create_all_extractors():
    """Create instances of all available extractors (17 total)."""
    return [
        # Spectral indices
        NDVIExtractor(),
        SAVIExtractor(),
        NDWIExtractor(),
        # Statistical extractors
        PCAExtractor(n_components=10),
        ICAExtractor(n_components=10),
        MNFExtractor(n_components=10),
        # Texture/Spatial extractors
        GLCMExtractor(distances=[1, 2]),
        LBPExtractor(radius=3, n_points=24),
        GaborExtractor(frequencies=[0.05, 0.1, 0.2]),
        MPExtractor(n_components=3, radii=[2, 4, 6]),
        # Wavelet extractors
        DWT1DExtractor(wavelet="db4", levels=3),
        DWT2DExtractor(wavelet="haar", levels=2),
        DWT3DExtractor(wavelet="haar", levels=1),
        # Moment extractors
        GeometricMomentExtractor(n_components=3, max_order=4),
        LegendreMomentExtractor(n_components=3, max_order=4),
        ZernikeMomentExtractor(n_components=3, max_order=6),
        # Other
        PPExtractor(n_projections=10),
    ]


def run_validation(
    dataset_name: str,
    dataset_path: Path,
    slurm_config: dict,
    num_jobs: int = 2,
    cores_per_job: int = 8,
) -> dict:
    """
    Run validation for a single dataset.

    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the H5 file
        slurm_config: SLURM configuration dictionary
        num_jobs: Number of SLURM jobs to spawn
        cores_per_job: Cores per SLURM job

    Returns:
        Dictionary with validation results
    """
    print(f"\n{'='*60}")
    print(f"Validating: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"SLURM config: {num_jobs} jobs × {cores_per_job} cores")
    print(f"{'='*60}")

    result = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "num_jobs": num_jobs,
        "cores_per_job": cores_per_job,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": None,
        "metrics": {},
    }

    try:
        # Load HSI data
        print(f"\n[1/4] Loading HSI data...")
        load_start = time.perf_counter()
        hsi = io.load_h5_hsi(str(dataset_path))
        load_time = time.perf_counter() - load_start
        print(f"      Loaded in {load_time:.2f}s")
        print(f"      Shape: {hsi.reflectance.shape}")
        result["metrics"]["load_time"] = load_time
        result["hsi_shape"] = list(hsi.reflectance.shape)

        # Create FeatureSpace with all extractors
        print(f"\n[2/4] Creating FeatureSpace with all extractors...")
        extractors = create_all_extractors()
        fs = FeatureSpace.from_list(extractors)
        print(f"      Extractors: {len(extractors)}")
        result["num_extractors"] = len(extractors)

        # Create SLURM runner
        print(f"\n[3/4] Initializing DaskSLURMRunner...")
        cluster_config = slurm_config["cluster"]
        runner = DaskSLURMRunner(
            cores=cores_per_job,
            memory=cluster_config.get("memory_per_job", "32GB"),
            processes=cluster_config.get("processes_per_job", 1),
            queue=cluster_config.get("queue", "normal"),
            walltime=cluster_config.get("walltime", "04:00:00"),
            num_jobs=num_jobs,
            account=cluster_config.get("account"),
            project=cluster_config.get("project"),
        )
        print(f"      Runner created successfully")

        # Run extraction
        print(f"\n[4/4] Running feature extraction...")
        extract_start = time.perf_counter()
        features = fs.extract(hsi, runner)
        extract_time = time.perf_counter() - extract_start
        print(f"      Extraction completed in {extract_time:.2f}s")
        result["metrics"]["extract_time"] = extract_time
        result["metrics"]["total_time"] = load_time + extract_time

        # Verify results
        print(f"\n[✓] Validation successful!")
        print(f"    Features extracted: {len(features)}")
        result["success"] = True
        result["num_features"] = len(features)
        result["feature_names"] = list(features.keys())

    except Exception as e:
        print(f"\n[✗] Validation failed: {e}")
        result["error"] = str(e)
        raise

    return result


def main():
    """Run validation for all datasets."""
    print("=" * 60)
    print("HYPPO SLURM Benchmark - Validation")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")

    # Load configuration
    datasets_config, slurm_config = load_config()

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Validation parameters
    num_jobs = 2
    cores_per_job = 8

    results = []
    datasets = datasets_config["datasets"]

    for dataset_name in datasets_config["execution_order"]:
        dataset_info = datasets[dataset_name]
        dataset_path = BENCHMARK_DIR / dataset_info["local_path"]

        if not dataset_path.exists():
            print(f"\n[SKIP] Dataset not found: {dataset_path}")
            results.append({
                "dataset": dataset_name,
                "success": False,
                "error": f"Dataset not found: {dataset_path}",
            })
            continue

        try:
            result = run_validation(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                slurm_config=slurm_config,
                num_jobs=num_jobs,
                cores_per_job=cores_per_job,
            )
            results.append(result)
        except Exception as e:
            results.append({
                "dataset": dataset_name,
                "success": False,
                "error": str(e),
            })

    # Save results
    output_file = RESULTS_DIR / f"validation_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for r in results:
        status = "✓ PASS" if r.get("success") else "✗ FAIL"
        print(f"  {r['dataset']}: {status}")
        if r.get("error"):
            print(f"    Error: {r['error']}")
        elif r.get("metrics"):
            print(f"    Time: {r['metrics'].get('total_time', 0):.2f}s")

    # Exit code
    all_success = all(r.get("success", False) for r in results)
    return 0 if all_success else 1


if __name__ == "__main__":
    exit(main())
