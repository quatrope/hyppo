import dask
import dask.threaded as dsk
from typing import Optional, Dict, List, Any
from .base import BaseRunner
from hyppo.core import HSI, FeatureResultCollection


class ThreadsRunner(BaseRunner):
    """
    ThreadsRunner using Dask.
    """

    def __init__(self, num_workers: Optional[int] = None) -> None:
        super().__init__()
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"Invalid number of workers: {num_workers}")
        self.num_workers = num_workers

    def resolve(self, data: HSI, feature_space) -> FeatureResultCollection:
        """
        Resolve feature extraction.

        Args:
            data: HSI object to process
            feature_space: FeatureSpace instance with feature graph
            execution_order: Topologically sorted list of extractor names

        Returns:
            FeatureResultCollection with extraction results
        """
        feature_graph = feature_space.feature_graph
        execution_order = feature_graph.get_execution_order()
        results = FeatureResultCollection()

        # Execute extractors in topological order
        for extractor_name in execution_order:
            extractor = feature_graph.extractors[extractor_name]
            input_mapping = feature_graph.get_input_mapping_for(extractor_name)

            # Prepare kwargs with dependency results
            input_kwargs = {}
            for input_name, source_name in input_mapping.items():
                if source_name in results:
                    # Pass the actual data, not the wrapped result
                    input_kwargs[input_name] = results[source_name].data

            # Add defaults for optional inputs not provided
            input_deps = extractor.get_input_dependencies()
            for input_name, dep_spec in input_deps.items():
                if input_name not in input_kwargs and not dep_spec.required:
                    # Try to get default
                    default_extractor = extractor.get_default_for_input(input_name)
                    if default_extractor is not None:
                        # Execute default extractor
                        try:
                            default_result = default_extractor.extract(data)
                            input_kwargs[input_name] = default_result
                        except TypeError:
                            # Fallback for extractors that don't support kwargs
                            default_result = default_extractor.extract(data)
                            input_kwargs[input_name] = default_result

            # Execute extractor with input kwargs
            result = extractor.extract(data, **input_kwargs)

            results.add_result(
                extractor_name=extractor_name,
                data=result,
                extractor=extractor,
                inputs_used=list(input_kwargs.keys()),
            )

        return results
