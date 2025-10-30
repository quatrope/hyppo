Advanced Usage
==============

This tutorial covers advanced features of HYPPO including complex extractor dependencies,
parallel execution with different runners, and distributed computing with SLURM.

Complex Extractor Dependencies
-------------------------------

Understanding Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some extractors depend on outputs from other extractors. HYPPO automatically
manages these dependencies:

.. code-block:: python

    from hyppo.core import FeatureSpace
    from hyppo.extractor import PCAExtractor, ICAExtractor
    
    # Create extractors with dependencies
    config = {
        "pca": (PCAExtractor(n_components=10), {}),
        # ICA depends on PCA output
        "ica": (ICAExtractor(n_components=5), {"pca_features": "pca"})
    }
    
    fs = FeatureSpace(config)
    
    # HYPPO resolves execution order automatically
    results = fs.extract(hsi)


Explicit Input Mapping
~~~~~~~~~~~~~~~~~~~~~~~

Define explicit input mappings for complex pipelines:

.. code-block:: python

    from hyppo.extractor import (
        MeanExtractor, StdExtractor, 
        PCAExtractor, CompositeExtractor
    )
    
    config = {
        # Base extractors (no dependencies)
        "mean": (MeanExtractor(), {}),
        "std": (StdExtractor(), {}),
        
        # PCA on original data
        "pca": (PCAExtractor(n_components=20), {}),
        
        # Composite extractor using outputs from mean, std, and pca
        "composite": (
            CompositeExtractor(),
            {
                "mean_input": "mean",
                "std_input": "std",
                "pca_input": "pca"
            }
        )
    }
    
    fs = FeatureSpace(config)


Dependency Graphs
~~~~~~~~~~~~~~~~~

Visualize the dependency graph to understand execution order:

.. code-block:: python

    # Get extractors in topologically sorted order
    extractors = fs.get_extractors()
    
    print("Execution order:")
    for idx, (name, extractor) in enumerate(extractors):
        deps = extractor.get_input_dependencies()
        if deps:
            print(f"{idx+1}. {name} (depends on: {list(deps.keys())})")
        else:
            print(f"{idx+1}. {name} (no dependencies)")


Parallel Execution
------------------

Sequential Runner (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default runner executes extractors sequentially:

.. code-block:: python

    from hyppo.runner import SequentialRunner
    
    # Explicitly use sequential runner
    runner = SequentialRunner()
    results = fs.extract(hsi, runner)


Thread-Based Parallel Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Dask with threads for I/O-bound tasks:

.. code-block:: python

    from hyppo.runner import DaskRunner
    
    # Create thread-based runner
    runner = DaskRunner.threads(num_threads=8)
    
    # Extract features in parallel
    results = fs.extract(hsi, runner)
    
    # Runner automatically handles dependencies and parallelizes
    # independent extractors


Process-Based Parallel Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use process-based parallelism for CPU-intensive tasks:

.. code-block:: python

    # Create process-based runner
    runner = DaskRunner.processes(
        num_workers=4,
        threads_per_worker=2
    )
    
    # Extract with process-based parallelism
    results = fs.extract(hsi, runner)
    
    # Best for CPU-bound extractors (PCA, GLCM, etc.)


Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

Compare different runner strategies:

.. code-block:: python

    import time
    from hyppo.runner import SequentialRunner, DaskRunner
    
    runners = {
        "sequential": SequentialRunner(),
        "threads_4": DaskRunner.threads(num_threads=4),
        "threads_8": DaskRunner.threads(num_threads=8),
        "processes_4": DaskRunner.processes(num_workers=4),
    }
    
    for name, runner in runners.items():
        start = time.time()
        results = fs.extract(hsi, runner)
        elapsed = time.time() - start
        print(f"{name}: {elapsed:.2f} seconds")


SLURM Distributed Computing
----------------------------

What is SLURM?
~~~~~~~~~~~~~~

SLURM (Simple Linux Utility for Resource Management) is a cluster management
system used in HPC environments. HYPPO can distribute feature extraction across
SLURM cluster nodes.

Basic SLURM Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a SLURM runner for cluster computing:

.. code-block:: python

    from hyppo.runner import DaskRunner
    
    # Create SLURM runner
    runner = DaskRunner.slurm(
        queue="normal",              # SLURM partition/queue name
        project="my_project",        # Account/project for billing
        cores=4,                     # Cores per worker
        memory="8GB",                # Memory per worker
        walltime="02:00:00",         # Maximum runtime (2 hours)
        n_workers=10                 # Number of SLURM jobs to spawn
    )
    
    # Extract features distributed across cluster
    results = fs.extract(hsi, runner)


Advanced SLURM Options
~~~~~~~~~~~~~~~~~~~~~~

Fine-tune SLURM job parameters:

.. code-block:: python

    runner = DaskRunner.slurm(
        queue="gpu",                 # GPU partition
        project="hyperspectral_01",
        cores=8,
        processes=2,                 # Processes per worker
        memory="16GB",
        walltime="04:00:00",
        n_workers=20,
        
        # Additional SLURM options
        job_extra=[
            "--gres=gpu:1",          # Request 1 GPU per job
            "--constraint=v100",     # Request specific GPU type
            "--exclusive",           # Exclusive node access
        ],
        
        # Dask-specific options
        local_directory="/scratch/dask-tmp",  # Temp directory on nodes
        silence_logs=False,          # Show detailed logs
    )


SLURM Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~

Create a configuration file for SLURM settings:

.. code-block:: yaml

    # slurm_config.yaml
    runner:
      type: slurm
      queue: normal
      project: hyperspectral_analysis
      cores: 4
      memory: 8GB
      walltime: "02:00:00"
      n_workers: 10
      job_extra:
        - "--exclusive"
        - "--mail-type=END"
        - "--mail-user=user@university.edu"

Load and use the configuration:

.. code-block:: python

    import yaml
    from hyppo.runner import DaskRunner
    
    # Load SLURM configuration
    with open("slurm_config.yaml") as f:
        config = yaml.safe_load(f)
    
    runner_cfg = config["runner"]
    runner = DaskRunner.slurm(**runner_cfg)


Batch Processing on SLURM
--------------------------

Processing Multiple HSI Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distribute multiple HSI files across cluster nodes:

.. code-block:: python

    from pathlib import Path
    from hyppo.runner import DaskRunner
    import hyppo
    
    # Setup SLURM runner
    runner = DaskRunner.slurm(
        queue="normal",
        project="hyperspectral",
        cores=4,
        memory="8GB",
        walltime="01:00:00",
        n_workers=20
    )
    
    # Load configuration
    fs = hyppo.io.load_config_yaml("pipeline.yaml")
    
    # Process all files
    input_dir = Path("hsi_data/")
    output_dir = Path("features/")
    output_dir.mkdir(exist_ok=True)
    
    for h5_file in input_dir.glob("*.h5"):
        print(f"Processing {h5_file.name}...")
        
        hsi = hyppo.io.load_h5_hsi(str(h5_file))
        results = fs.extract(hsi, runner)
        
        output_file = output_dir / f"{h5_file.stem}_features.h5"
        results.save(str(output_file))


SLURM Job Script
~~~~~~~~~~~~~~~~

Create a SLURM batch script for automated processing:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=hyppo_extraction
    #SBATCH --partition=normal
    #SBATCH --time=04:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16GB
    #SBATCH --output=hyppo_%j.out
    #SBATCH --error=hyppo_%j.err
    
    # Load environment
    module load python/3.11
    source venv/bin/activate
    
    # Run extraction
    python extraction_script.py --config pipeline.yaml --input hsi_data/ --output features/

Create the Python script ``extraction_script.py``:

.. code-block:: python

    import argparse
    from pathlib import Path
    import hyppo
    from hyppo.runner import DaskRunner
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--input", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()
        
        # Load configuration
        fs = hyppo.io.load_config_yaml(args.config)
        
        # Setup SLURM runner
        runner = DaskRunner.slurm(
            queue="normal",
            cores=4,
            memory="8GB",
            walltime="01:00:00",
            n_workers=10
        )
        
        # Process files
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        for h5_file in input_dir.glob("*.h5"):
            hsi = hyppo.io.load_h5_hsi(str(h5_file))
            results = fs.extract(hsi, runner)
            
            output_file = output_dir / f"{h5_file.stem}_features.h5"
            results.save(str(output_file))
    
    if __name__ == "__main__":
        main()


Monitoring and Debugging
-------------------------

Monitor SLURM Jobs
~~~~~~~~~~~~~~~~~~

Check SLURM job status:

.. code-block:: bash

    # Check job status
    squeue -u $USER
    
    # View job details
    scontrol show job <job_id>
    
    # Check job output
    tail -f hyppo_<job_id>.out


Dask Dashboard
~~~~~~~~~~~~~~

Access the Dask dashboard for real-time monitoring:

.. code-block:: python

    from hyppo.runner import DaskRunner
    
    runner = DaskRunner.slurm(
        queue="normal",
        cores=4,
        memory="8GB",
        walltime="02:00:00",
        n_workers=10,
        
        # Enable dashboard
        scheduler_options={"dashboard_address": ":8787"}
    )
    
    # Dashboard available at http://<scheduler-node>:8787


Debug Failed Extractions
~~~~~~~~~~~~~~~~~~~~~~~~~

Handle errors gracefully in distributed settings:

.. code-block:: python

    from hyppo.runner import DaskRunner
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    runner = DaskRunner.slurm(
        queue="normal",
        cores=4,
        memory="8GB",
        walltime="02:00:00",
        n_workers=10,
        silence_logs=False  # Show detailed logs
    )
    
    try:
        results = fs.extract(hsi, runner)
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        # Handle error (retry, skip, etc.)


Best Practices
--------------

1. **Start small**: Test on small datasets before scaling to cluster
2. **Use checkpoints**: Save intermediate results to avoid re-computation
3. **Monitor resources**: Check CPU, memory, and I/O usage
4. **Profile first**: Identify bottlenecks before parallelizing
5. **Balance workers**: Don't over-subscribe cluster resources
6. **Handle failures**: Implement retry logic for transient errors
7. **Clean up**: Close runners and delete temporary files
8. **Log everything**: Keep detailed logs for debugging

Performance Optimization Tips
------------------------------

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Process in chunks for large HSI
    # Use memory-efficient extractors
    # Clear intermediate results
    
    results = fs.extract(hsi, runner)
    results.save("output.h5")
    
    # Free memory
    del hsi
    del results


Choosing the Right Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Sequential**: Small HSI, few extractors, debugging
- **Threads**: I/O-bound tasks, shared memory needed
- **Processes**: CPU-bound tasks, independent extractors
- **SLURM**: Very large HSI, many files, cluster available

Next Steps
----------

- Review :doc:`basic_usage` for fundamentals
- Check :doc:`config` for pipeline management
- Explore :doc:`hsi_io` for data handling
