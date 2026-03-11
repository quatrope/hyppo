"""Dask-based parallel runner for feature extraction."""

from typing import Iterable

from dask.distributed import Client, LocalCluster

from hyppo.core import Feature, FeatureCollection, HSI
from .base import BaseRunner


class DaskRunner(BaseRunner):
    """
    Base Dask runner with shared execution logic.

    This class contains the core Dask graph building and execution logic
    shared by all Dask-based runners. It should not be instantiated directly;
    instead use one of the concrete runner classes:
        - DaskThreadsRunner: Thread-based parallelism
        - DaskProcessesRunner: Process-based parallelism
    """

    def __init__(self, client: Client):
        """
        Initialize DaskRunner with a Dask client.

        Args:
            client: Configured Dask Client instance
        """
        super().__init__()
        self._client = client
        self._cluster = client.cluster

    def resolve(self, data: HSI, feature_space) -> FeatureCollection:
        """Resolve feature extraction using a complete Dask graph.

        Uses distributed execution.

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
        computed_results: Iterable = self._client.get(
            dask_graph, result_keys
        )  # type: ignore

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

        The graph structure follows Dask's requirements where each
        task is defined as:
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
                # Each dependency becomes a separate argument
                # Dask resolves these automatically before calling
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
    assert len(args) >= 2

    dependency_results = args[:-2]
    input_names = args[-2]
    defaults = args[-1]

    input_kwargs = _build_kwargs_from_dependencies(
        input_names, dependency_results
    )
    _apply_default_extractors(input_kwargs, defaults, hsi_data)

    return extractor.extract(hsi_data, **input_kwargs)


def _build_kwargs_from_dependencies(input_names, dependency_results):
    """Map dependency results to their input names."""
    input_kwargs = {}
    for i, input_name in enumerate(input_names):
        if i < len(dependency_results):
            input_kwargs[input_name] = dependency_results[i]
    return input_kwargs


def _apply_default_extractors(input_kwargs, defaults, hsi_data):
    """Fill missing optional inputs with default extractor results."""
    for input_name, default_extractor in defaults.items():
        if input_name not in input_kwargs:
            default_result = default_extractor.extract(hsi_data)
            input_kwargs[input_name] = default_result


class DaskThreadsRunner(DaskRunner):
    """
    Thread-based Dask runner for parallel feature extraction.

    Uses a single worker with multiple threads for parallel execution.
    Suitable for I/O-bound tasks or when memory sharing is beneficial.
    """

    def __init__(self, num_threads: int | None = None):
        """
        Create a thread-based Dask runner.

        Args:
            num_threads: Number of threads to use
                        (None = use all available cores)

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
        super().__init__(client)


class DaskProcessesRunner(DaskRunner):
    """
    Process-based Dask runner for parallel feature extraction.

    Uses multiple worker processes for true parallel execution.
    Suitable for CPU-bound tasks that can benefit from multiple cores.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        threads_per_worker: int = 1,
        memory_limit: str = "auto",
    ):
        """
        Create a process-based Dask runner.

        Args:
            num_workers: Number of worker processes
                        (None = use all available cores)
            threads_per_worker: Number of threads per worker process
            memory_limit: Memory limit per worker (e.g., "2GB", "auto")

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
        super().__init__(client)
