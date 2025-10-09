from hyppo.core import HSI, FeatureResultCollection
from .base import BaseRunner


class SequentialRunner(BaseRunner):
    """
    Sequential runner that executes extractors one by one in topological order.

    This is the simplest runner implementation that processes extractors
    sequentially in the same process without any parallelization.
    """

    def resolve(self, data: HSI, feature_space) -> FeatureResultCollection:
        """
        Resolve feature extraction sequentially.

        Args:
            data: HSI object to process
            feature_space: FeatureSpace instance with feature graph

        Returns:
            FeatureResultCollection with extraction results
        """
        feature_graph = feature_space.feature_graph
        results = FeatureResultCollection({})
        extracted_results = {}

        for extractor_name in feature_graph.get_execution_order():
            extractor = feature_graph.extractors[extractor_name]
            input_mapping = feature_graph.get_input_mapping_for(extractor_name)

            input_kwargs = {}
            for input_name, source_name in input_mapping.items():
                input_kwargs[input_name] = extracted_results[source_name]

            defaults = self._get_defaults_for_extractor(extractor)
            for input_name, default_extractor in defaults.items():
                if input_name not in input_kwargs:
                    default_result = default_extractor.extract(data)
                    input_kwargs[input_name] = default_result

            result = extractor.extract(data, **input_kwargs)

            extracted_results[extractor_name] = result

            results.add_result(
                extractor_name=extractor_name,
                data=result,
                extractor=extractor,
                inputs_used=list(input_mapping.keys()),
            )

        return results
