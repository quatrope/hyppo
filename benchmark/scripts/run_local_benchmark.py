#!/usr/bin/env python3
"""
Benchmark script using LocalProcessRunner.

Runs on a SLURM node with local multiprocessing parallelization.
Submit with: sbatch submit_benchmark.sh
"""

import json
import os
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
from hyppo.runner import LocalProcessRunner

BENCHMARK_DIR = Path(__file__).parent.parent
CONFIG_DIR = BENCHMARK_DIR / "config"
DATA_DIR = BENCHMARK_DIR / "data"
RESULTS_DIR = BENCHMARK_DIR / "results" / "validation"


def load_config():
    """Load benchmark configuration files."""
    with open(CONFIG_DIR / "datasets.yaml") as f:
        datasets_config = yaml.safe_load(f)
    return datasets_config


def create_all_extractors():
    """Create instances of all available extractors (17 total)."""
    return [
        NDVIExtractor(),
        SAVIExtractor(),
        NDWIExtractor(),
        PCAExtractor(n_components=10),
        ICAExtractor(n_components=10),
        MNFExtractor(n_components=10),
        # GLCMExtractor(distances=[1, 2]),
        # LBPExtractor(radius=3, n_points=24),
        # GaborExtractor(frequencies=[0.05, 0.1, 0.2]),
        MPExtractor(n_components=3, radii=[2, 4, 6]),
        DWT1DExtractor(wavelet="db4", levels=3),
        DWT2DExtractor(wavelet="haar", levels=2),
        DWT3DExtractor(wavelet="haar", levels=1),
        # GeometricMomentExtractor(n_components=3, max_order=4),
        # LegendreMomentExtractor(n_components=3, max_order=4),
        # ZernikeMomentExtractor(n_components=3, max_order=6),
        PPExtractor(n_projections=10),
    ]


def run_benchmark(dataset_name: str, dataset_path: Path, num_workers: int) -> dict:
    """Run benchmark for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}")

    result = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "num_workers": num_workers,
        "node": os.environ.get("SLURMD_NODENAME", "local"),
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": None,
        "metrics": {},
    }

    try:
        # Load HSI data
        print(f"\n[1/3] Loading HSI data...")
        load_start = time.perf_counter()
        hsi = io.load_h5_hsi(str(dataset_path))
        load_time = time.perf_counter() - load_start
        print(f"      Shape: {hsi.reflectance.shape}")
        print(f"      Time: {load_time:.2f}s")
        result["metrics"]["load_time"] = load_time
        result["hsi_shape"] = list(hsi.reflectance.shape)

        # Create FeatureSpace
        print(f"\n[2/3] Creating FeatureSpace...")
        extractors = create_all_extractors()
        fs = FeatureSpace.from_list(extractors)
        print(f"      Extractors: {len(extractors)}")
        for ext in extractors:
            print(f"        - {ext.__class__.__name__}")
        result["num_extractors"] = len(extractors)

        # Run extraction with LocalProcessRunner
        print(f"\n[3/3] Running extraction with {num_workers} workers...")
        runner = LocalProcessRunner(num_workers=num_workers)
        extract_start = time.perf_counter()
        features = fs.extract(hsi, runner)
        extract_time = time.perf_counter() - extract_start
        print(f"      Time: {extract_time:.2f}s")

        result["metrics"]["extract_time"] = extract_time
        result["metrics"]["total_time"] = load_time + extract_time
        result["success"] = True
        result["num_features"] = len(features)
        result["feature_names"] = list(features.keys())

        print(f"\n[OK] Extracted: {list(features.keys())}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


def main():
    """Run benchmark."""
    print("=" * 60)
    print("HYPPO Benchmark - LocalProcessRunner")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'local')}")
    print(f"CPUs available: {os.cpu_count()}")
    print(f"SLURM CPUs: {os.environ.get('SLURM_CPUS_PER_TASK', 'N/A')}")

    # Determine number of workers
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    print(f"Using {num_workers} workers")

    # Load config
    datasets_config = load_config()

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    datasets = datasets_config["datasets"]

    for dataset_name in datasets_config["execution_order"]:
        dataset_info = datasets[dataset_name]
        dataset_path = BENCHMARK_DIR / dataset_info["local_path"]

        if not dataset_path.exists():
            print(f"\n[SKIP] Not found: {dataset_path}")
            results.append({
                "dataset": dataset_name,
                "success": False,
                "error": f"Dataset not found",
            })
            continue

        result = run_benchmark(dataset_name, dataset_path, num_workers)
        results.append(result)

    # Save results
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_file = RESULTS_DIR / f"benchmark_{job_id}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        if r.get("metrics"):
            time_str = f"{r['metrics']['total_time']:.2f}s"
        else:
            time_str = r.get("error", "N/A")
        print(f"  {r['dataset']}: {status} ({time_str})")

    print(f"\nResults: {output_file}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
