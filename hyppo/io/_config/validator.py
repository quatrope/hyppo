from hyppo.extractor import registry
from .models import Config, PipelineConfig, ExtractorConfig


class ConfigValidator:
    """Validator for HYPPO configuration objects."""

    def validate(self, config: Config) -> None:
        """
        Validate complete configuration.

        Args:
            config: Configuration object to validate

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If configuration is invalid
        """
        self._validate_pipeline(config.pipeline)

    def _validate_pipeline(self, pipeline: PipelineConfig) -> None:
        """Validate pipeline configuration."""
        if not pipeline.extractors:
            raise ValueError("Pipeline must contain at least one extractor")

        # Validate individual extractors
        for name, extractor_config in pipeline.extractors.items():
            self._validate_extractor(name, extractor_config, pipeline)

        # Validate dependency graph (no cycles)
        self._validate_dependency_graph(pipeline)

    def _validate_extractor(
        self, name: str, extractor_config: ExtractorConfig, pipeline: PipelineConfig
    ) -> None:
        """Validate single extractor configuration."""
        # Check if extractor type exists
        extractor_type = extractor_config.extractor_type
        if not registry.is_registered(extractor_type):
            available_names = list(registry.list_extractors())
            raise ValueError(
                f"Unknown extractor type: '{extractor_type}'. "
                f"Available extractors: {available_names}"
            )

        # Validate input dependencies
        for input_name, source_name in extractor_config.inputs.items():
            if source_name not in pipeline.extractors:
                raise ValueError(
                    f"Extractor '{name}' depends on '{source_name}' "
                    f"which is not defined in pipeline"
                )

    def _validate_dependency_graph(self, pipeline: PipelineConfig) -> None:
        """Validate that dependency graph has no cycles."""
        # Build adjacency list
        graph = {}
        for name, extractor_config in pipeline.extractors.items():
            graph[name] = list(extractor_config.inputs.values())

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError(
                        "Circular dependency detected in pipeline. "
                        "Extractors cannot depend on themselves or form cycles."
                    )
