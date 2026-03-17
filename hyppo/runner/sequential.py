"""Sequential runner for single-threaded feature extraction."""

from hyppo.core import Feature, FeatureCollection, HSI
from .base import BaseRunner


class SequentialRunner(BaseRunner):
    """
    Sequential runner that executes extractors one by one in topological order.

    This is the simplest runner implementation that processes extractors
    sequentially in the same process without any parallelization.
    """

    def resolve(self, data: HSI, feature_space) -> FeatureCollection:
        """
        Resolve feature extraction sequentially.

        Parameters
        ----------
        data : HSI
            HSI object to process
        feature_space : FeatureSpace
            FeatureSpace instance with feature graph

        Returns
        -------
        FeatureCollection
            FeatureCollection with extraction results
        """
        feature_graph = feature_space.feature_graph
        results = {}
        extracted_results = {}

        for extractor_name in feature_graph.get_execution_order():
            extractor = feature_graph.extractors[extractor_name]
            input_mapping = feature_graph.get_input_mapping_for(extractor_name)

            input_kwargs = self._build_input_kwargs(
                input_mapping, extracted_results, extractor, data
            )

            result = extractor.extract(data, **input_kwargs)

            extracted_results[extractor_name] = result

            results[extractor_name] = Feature(
                result, extractor, list(input_mapping.keys())
            )

        return FeatureCollection.from_features(results)

    def _build_input_kwargs(
        self, input_mapping, extracted_results, extractor, data
    ):
        """Build input keyword arguments for an extractor."""
        input_kwargs = {}
        for input_name, source_name in input_mapping.items():
            input_kwargs[input_name] = extracted_results[source_name]

        defaults = self._get_defaults_for_extractor(extractor)
        for input_name, default_extractor in defaults.items():
            if input_name not in input_kwargs:
                default_result = default_extractor.extract(data)
                input_kwargs[input_name] = default_result

        return input_kwargs
