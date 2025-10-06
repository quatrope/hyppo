from typing import Iterable
from .base import BaseRunner
from hyppo.core import HSI, FeatureResultCollection
from dask.distributed import Client, LocalCluster


class DaskRunner(BaseRunner):
    """
    Dask-based runner for parallel feature extraction.

    Provides functionality for Dask distributed execution including
    cluster management, graph building, and resource cleanup.
    """

    def __init__(self, client: Client):
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
            raise ValueError(f"Invalid threads per worker: {threads_per_worker}")

        cluster = LocalCluster(
            n_workers=num_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=memory_limit,
            silence_logs=True,
        )
        client = Client(cluster)
        return cls(client)

    def resolve(self, data: HSI, feature_space) -> FeatureResultCollection:
        """
        Resolve feature extraction using a complete Dask graph with distributed execution.

        Args:
            data: HSI object to process
            feature_space: FeatureSpace instance with feature graph

        Returns:
            FeatureResultCollection with extraction results
        """

        feature_graph = feature_space.feature_graph

        # Build complete Dask computation graph
        dask_graph = self._build_dask_graph(data, feature_graph)

        # Get all final result keys (extractor names)
        result_keys = list(feature_graph.extractors.keys())

        # Execute the entire graph with distributed scheduler
        computed_results: Iterable = self._client.get(dask_graph, result_keys)  # type: ignore

        # Build FeatureResultCollection from results
        results = FeatureResultCollection()
        for extractor_name, result_data in zip(result_keys, computed_results):
            if result_data is not None:
                extractor = feature_graph.extractors[extractor_name]
                input_mapping = feature_graph.get_input_mapping_for(extractor_name)

                results.add_result(
                    extractor_name=extractor_name,
                    data=result_data,
                    extractor=extractor,
                    inputs_used=list(input_mapping.keys()),
                )

        return results

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
            task_args.append(self._get_defaults_for_extractor(extractor))  # defaults

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
