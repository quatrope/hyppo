from abc import ABC, abstractmethod
from .base import BaseRunner
from hyppo.core import HSI, FeatureResultCollection
from dask.distributed import LocalCluster, Client


class DaskRunner(BaseRunner, ABC):
    """
    Abstract base class for Dask-based runners.
    
    Provides common functionality for Dask distributed execution including
    cluster management, graph building, and resource cleanup.
    """

    _cluster: LocalCluster | None
    _client: Client | None

    def __init__(self) -> None:
        super().__init__()
        self._cluster = None
        self._client = None

    def resolve(self, data: HSI, feature_space) -> FeatureResultCollection:
        """
        Resolve feature extraction using a complete Dask graph with distributed execution.

        Args:
            data: HSI object to process
            feature_space: FeatureSpace instance with feature graph

        Returns:
            FeatureResultCollection with extraction results
        """
        self._setup_cluster()
        
        try:
            feature_graph = feature_space.feature_graph
            
            # Build complete Dask computation graph
            dask_graph = self._build_dask_graph(data, feature_graph)
            
            # Get all final result keys (extractor names)
            result_keys = list(feature_graph.extractors.keys())
            
            # Execute the entire graph with distributed scheduler
            assert self._client is not None
            computed_results = self._client.get(dask_graph, result_keys)
            
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
            
        finally:
            # Cleanup cluster
            self._cleanup_cluster()

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
        hsi_key = 'hsi_data'
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
    
    def _get_defaults_for_extractor(self, extractor) -> dict:
        """
        Get default extractors for optional inputs.

        Args:
            extractor: The extractor to get defaults for
            
        Returns:
            Dictionary mapping input names to default extractors
        """
        defaults = {}
        input_deps = extractor.get_input_dependencies()
        
        for input_name, dep_spec in input_deps.items():
            if not dep_spec.required:
                default_extractor = extractor.get_default_for_input(input_name)
                if default_extractor is not None:
                    defaults[input_name] = default_extractor
                    
        return defaults

    @abstractmethod
    def _setup_cluster(self):
        """Setup Dask distributed cluster. Must be implemented by subclasses."""
        pass

    def _cleanup_cluster(self):
        """Cleanup Dask distributed cluster."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None

    def __enter__(self):
        """Context manager entry."""
        self._setup_cluster()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup_cluster()


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
    if len(args) >= 2:
        dependency_results = args[:-2]
        input_names = args[-2]
        defaults = args[-1]
    else:
        dependency_results = []
        input_names = args[0] if len(args) > 0 else []
        defaults = args[1] if len(args) > 1 else {}
    
    # Build input kwargs from dependency results
    input_kwargs = {}
    for i, input_name in enumerate(input_names):
        if i < len(dependency_results):
            input_kwargs[input_name] = dependency_results[i]
    
    # Add defaults for optional inputs not provided
    for input_name, default_extractor in defaults.items():
        if input_name not in input_kwargs:
            try:
                default_result = default_extractor.extract(hsi_data)
                input_kwargs[input_name] = default_result
            except TypeError:
                # Fallback for extractors that don't support kwargs
                default_result = default_extractor.extract(hsi_data)
                input_kwargs[input_name] = default_result
    
    # Execute extractor with resolved inputs
    return extractor.extract(hsi_data, **input_kwargs)