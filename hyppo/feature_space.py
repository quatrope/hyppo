from typing import Dict, Tuple, Any
from hyppo.extractor.base import Extractor
from hyppo.feature_dependency_graph import FeatureDependencyGraph
from hyppo.hsi import HSI


class FeatureSpace:

    def __init__(self, extractor_configs: Dict[str, Tuple[Extractor, Dict[str, str]]]):
        """
        Initialize FeatureSpace with feature extractor configurations.

        Args:
            extractor_configs: Dict of {name: (extractor, input_mapping)}
                             where input_mapping is {input_name: source_extractor_name}
        """
        self.extractor_configs = extractor_configs
        self.extractors = {
            name: config[0] for name, config in extractor_configs.items()
        }
        self.feature_graph = self._build_feature_dependency_graph()

    def _build_feature_dependency_graph(self):
        """Build and validate the feature dependency graph."""
        graph = FeatureDependencyGraph()

        for name, (extractor, input_mapping) in self.extractor_configs.items():
            graph.add_extractor(name, extractor, input_mapping)

        graph.validate()
        return graph

    def extract(self, data: HSI, runner=None):
        """
        Extract features with dependency resolution.

        Args:
            data: HSI object to extract features from
            runner: Runner to execute extraction (defaults to ThreadsRunner)
        """
        if runner is None:
            runner = self.get_default_runner()

        return runner.resolve(data, self)

    def get_extractors(self):
        """Get all extractors."""
        return self.extractors

    def get_default_runner(self):
        """Get the default runner."""
        from hyppo.runner.threads import ThreadsRunner

        return ThreadsRunner()
