"""Dask-based parallel runner for feature extraction."""

from typing import Iterable

from dask.distributed import Client, LocalCluster

from hyppo.core import Feature, FeatureCollection, HSI
from .base import BaseRunner

try:
    from dask_jobqueue import SLURMCluster
except ImportError:
    SLURMCluster = None


class DaskRunner(BaseRunner):
    """
    Dask-based runner for parallel feature extraction.

    Provides functionality for Dask distributed execution across different
    backend types: thread-based, process-based (local), and SLURM cluster-based.
    Includes cluster management, graph building, and resource cleanup.

    Factory methods:
        - threads(): Thread-based parallelism on local machine
        - processes(): Process-based parallelism on local machine
        - slurm(): SLURM cluster-based distributed execution (requires dask-jobqueue)
    """

    def __init__(self, client: Client):
        """Initialize DaskRunner with a Dask client."""
        super().__init__()
        self._client = client
        self._cluster = client.cluster

    @classmethod
    def threads(cls, num_threads: int | None = None):
        """
        Create a DaskRunner configured for thread-based parallel execution.

        Args:
            num_threads: Number of threads to use (None = use all available cores)

        Returns:
            DaskRunner instance configured for thread-based execution

        Raises:
            ValueError: If num_threads is less than 1
        """
        if num_threads is not None and num_threads < 1:
            raise ValueError(f"Invalid number of threads: {num_threads}")

        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=num_threads,
            processes=False,
            memory_limit="auto",
            silence_logs=True,
        )
        client = Client(cluster)
        return cls(client)

    @classmethod
    def processes(
        cls,
        num_workers: int | None = None,
        threads_per_worker: int = 1,
        memory_limit: str = "auto",
    ):
        """
        Create a DaskRunner configured for process-based parallel execution.

        Args:
            num_workers: Number of worker processes (None = use all available cores)
            threads_per_worker: Number of threads per worker process
            memory_limit: Memory limit per worker (e.g., "2GB", "auto")

        Returns:
            DaskRunner instance configured for process-based execution

        Raises:
            ValueError: If num_workers or threads_per_worker is less than 1
        """
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"Invalid number of workers: {num_workers}")
        if threads_per_worker < 1:
            raise ValueError(
                f"Invalid threads per worker: {threads_per_worker}"
            )

        cluster = LocalCluster(
            n_workers=num_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=memory_limit,
            silence_logs=True,
        )
        client = Client(cluster)
        return cls(client)

    @classmethod
    def slurm(
        cls,
        cores: int = 1,
        memory: str = "4GB",
        processes: int = 1,
        queue: str = "normal",
        walltime: str = "01:00:00",
        num_jobs: int = 1,
        account: str | None = None,
        project: str | None = None,
        job_extra_directives: list[str] | None = None,
        **kwargs,
    ):
        """
        Create a DaskRunner configured for SLURM cluster execution.

        Must be run on a system with SLURM installed (HPC cluster login node).
        Creates SLURM jobs that serve as Dask workers.

        Args:
            cores: Number of cores per SLURM job
            memory: Memory per SLURM job (e.g., "4GB", "16GB")
            processes: Number of worker processes per job
            queue: SLURM queue/partition name
            walltime: Maximum job duration (HH:MM:SS format)
            num_jobs: Number of SLURM jobs to spawn
            account: SLURM account to charge
            project: SLURM project name
            job_extra_directives: Additional SBATCH directives as list of strings
                                 (e.g., ["--constraint=haswell", "--exclusive"])
            **kwargs: Additional keyword arguments passed to SLURMCluster

        Returns:
            DaskRunner instance configured for SLURM execution

        Raises:
            ImportError: If dask-jobqueue is not installed
            ValueError: If cores, processes, or num_jobs is less than 1

        Example:
            >>> runner = DaskRunner.slurm(
            ...     cores=8,
            ...     memory="32GB",
            ...     processes=1,
            ...     queue="normal",
            ...     walltime="02:00:00",
            ...     num_jobs=10,
            ...     account="my_project",
            ...     job_extra_directives=["--constraint=haswell"]
            ... )
            >>> results = feature_space.extract(hsi, runner)
        """
        if SLURMCluster is None:
            raise ImportError(
                "dask-jobqueue is required for SLURM execution. "
                "Install with: pip install dask-jobqueue"
            )

        if cores < 1:
            raise ValueError(f"Invalid number of cores: {cores}")
        if processes < 1:
            raise ValueError(f"Invalid number of processes: {processes}")
        if num_jobs < 1:
            raise ValueError(f"Invalid number of jobs: {num_jobs}")

        # Build cluster configuration
        cluster_kwargs = {
            "cores": cores,
            "memory": memory,
            "processes": processes,
            "queue": queue,
            "walltime": walltime,
            "silence_logs": True,
        }

        # Add optional parameters
        if account is not None:
            cluster_kwargs["account"] = account
        if project is not None:
            cluster_kwargs["project"] = project
        if job_extra_directives is not None:
            cluster_kwargs["job_extra_directives"] = job_extra_directives

        # Merge any additional kwargs
        cluster_kwargs.update(kwargs)

        # Create SLURM cluster
        cluster = SLURMCluster(**cluster_kwargs)

        # Scale to requested number of jobs
        cluster.scale(jobs=num_jobs)

        # Create client
        client = Client(cluster)

        return cls(client)

    def resolve(self, data: HSI, feature_space) -> FeatureCollection:
        """Resolve feature extraction using a complete Dask graph with distributed execution.

        Args:
            data: HSI object to process
            feature_space: FeatureSpace instance with feature graph

        Returns:
            FeatureCollection with extraction results
        """
        feature_graph = feature_space.feature_graph

        # Build complete Dask computation graph
        dask_graph = self._build_dask_graph(data, feature_graph)

        # Get all final result keys (extractor names)
        result_keys = list(feature_graph.extractors.keys())

        # Execute the entire graph with distributed scheduler
        computed_results: Iterable = self._client.get(dask_graph, result_keys)  # type: ignore

        # Build FeatureCollection from results
        results = {}
        for extractor_name, result_data in zip(result_keys, computed_results):
            if result_data is not None:
                extractor = feature_graph.extractors[extractor_name]
                input_mapping = feature_graph.get_input_mapping_for(
                    extractor_name
                )

                results[extractor_name] = Feature(
                    result_data, extractor, list(input_mapping.keys())
                )

        return FeatureCollection.from_features(results)

    def _build_dask_graph(self, data: HSI, feature_graph) -> dict:
        """
        Build a complete Dask computation graph for all extractors.

        The graph structure follows Dask's requirements where each task is defined as:
        key: (function, arg1, arg2, ..., argN)

        Dependencies are expressed by referencing other task keys as arguments.

        Args:
            data: HSI object to process
            feature_graph: Feature dependency graph

        Returns:
            Dictionary representing the Dask computation graph
        """
        graph = {}

        # Add HSI data as a constant in the graph
        hsi_key = "hsi_data"
        graph[hsi_key] = data

        # Process each extractor to build the computation graph
        for extractor_name in feature_graph.extractors.keys():
            extractor = feature_graph.extractors[extractor_name]
            input_mapping = feature_graph.get_input_mapping_for(extractor_name)

            # Start building task arguments list
            task_args = [_execute_extractor_task, extractor, hsi_key]

            # Add dependency results as direct references to other task keys
            for input_name, source_name in input_mapping.items():
                # Each dependency becomes a separate argument to the function
                # Dask will resolve these automatically before calling the function
                task_args.append(source_name)

            # Add metadata about input names and defaults
            task_args.append(list(input_mapping.keys()))  # input names
            task_args.append(
                self._get_defaults_for_extractor(extractor)
            )  # defaults

            # Create the task tuple for this extractor
            graph[extractor_name] = tuple(task_args)

        return graph


def _execute_extractor_task(extractor, hsi_data, *args):
    """
    Execute a single extractor task within the Dask graph.

    This function receives resolved dependency results as positional arguments
    followed by metadata about input names and defaults.

    Args:
        extractor: The extractor instance to execute
        hsi_data: HSI data object
        *args: Variable arguments containing:
               - dependency results (in order of input_mapping)
               - input_names list
               - defaults dict

    Returns:
        Extraction results from the extractor
    """
    # Split args into dependency results and metadata
    # Last two args are always input_names and defaults
    assert len(args) >= 2

    dependency_results = args[:-2]
    input_names = args[-2]
    defaults = args[-1]

    # Build input kwargs from dependency results
    input_kwargs = {}
    for i, input_name in enumerate(input_names):
        if i < len(dependency_results):
            input_kwargs[input_name] = dependency_results[i]

    # Add defaults for optional inputs not provided
    for input_name, default_extractor in defaults.items():
        if input_name not in input_kwargs:
            default_result = default_extractor.extract(hsi_data)
            input_kwargs[input_name] = default_result

    # Execute extractor with resolved inputs
    return extractor.extract(hsi_data, **input_kwargs)
