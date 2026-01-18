from typing import TYPE_CHECKING

from .dependency_graph import FeatureDependencyGraph
from .._hsi import HSI

if TYPE_CHECKING:
    from hyppo.extractor.base import Extractor
    from hyppo.runner import BaseRunner


class FeatureSpace:
    """Orchestrates feature extraction with dependency management."""

    def __init__(self, extractor_configs):
        """
        Initialize FeatureSpace with feature extractor configurations.

        Args:
            extractor_configs: Dict of {name: (extractor, input_mapping)}
        """
        self.extractor_configs = extractor_configs
        self.extractors = {
            name: config[0] for name, config in extractor_configs.items()
        }
        self.feature_graph = self._build_feature_dependency_graph()

    def extract(self, data: HSI, runner: "BaseRunner | None" = None):
        """
        Extract features with dependency resolution.

        Args:
            data: HSI object to extract features from
            runner: Runner to execute extraction (defaults to ThreadsRunner)
        """
        if runner is None:
            runner = self._get_default_runner()

        return runner.resolve(data, self)

    def get_extractors(self):
        """Get all extractors."""
        return self.extractors

    def _get_default_runner(self):
        """Get the default runner."""
        from hyppo.runner import SequentialRunner

        return SequentialRunner()

    def _build_feature_dependency_graph(self):
        """Build and validate the feature dependency graph."""
        graph = FeatureDependencyGraph()

        for name, (extractor, input_mapping) in self.extractor_configs.items():
            graph.add_extractor(name, extractor, input_mapping)

        graph.validate()
        return graph

    @classmethod
    def from_list(cls, extractors: list["Extractor"]):
        """
        Convert list of extractors to configuration dict.

        Performs automatic dependency resolution.

        Args:
            extractors: List of extractor instances

        Returns:
            Dict configuration with automatic dependency mappings

        Raises:
            ValueError: If duplicates or missing dependencies found
        """
        if not extractors:
            return cls({})

        # Create name-to-extractor mapping and check for duplicates
        extractor_mapping = {}
        type_to_name = {}

        for extractor in extractors:
            extractor_name = extractor.feature_name()
            extractor_type = type(extractor)

            # Check for duplicate names
            if extractor_name in extractor_mapping:
                msg = (
                    f"Duplicate extractor name '{extractor_name}'. "
                    f"Cannot have multiple extractors of same type."
                )
                raise ValueError(msg)

            extractor_mapping[extractor_name] = extractor
            type_to_name[extractor_type] = extractor_name

        # Resolve dependencies for each extractor
        config_dict = {}

        for extractor_name, extractor in extractor_mapping.items():
            input_dependencies = extractor.get_input_dependencies()
            input_mapping = {}

            # Resolve each declared dependency
            for input_name, dep_spec in input_dependencies.items():
                required_type = dep_spec["extractor"]

                # Look for an extractor of the required type
                matching_extractor_name = None
                for candidate_name, cand in extractor_mapping.items():
                    if isinstance(cand, required_type):
                        if matching_extractor_name is not None:
                            # Multiple matches - ambiguous
                            msg = (
                                f"Ambiguous dependency: extractor "
                                f"'{extractor_name}' requires input "
                                f"'{input_name}' of type "
                                f"'{required_type.__name__}', but found "
                                f"multiple candidates: "
                                f"'{matching_extractor_name}' and "
                                f"'{candidate_name}'"
                            )
                            raise ValueError(msg)
                        matching_extractor_name = candidate_name

                if matching_extractor_name is not None:
                    input_mapping[input_name] = matching_extractor_name
                elif dep_spec["required"]:
                    msg = (
                        f"Missing required dependency: extractor "
                        f"'{extractor_name}' requires input "
                        f"'{input_name}' of type "
                        f"'{required_type.__name__}', but no such "
                        f"extractor found in the list"
                    )
                    raise ValueError(msg)

            config_dict[extractor_name] = (extractor, input_mapping)

        return cls(config_dict)

    def save_config(self, path) -> None:
        """
        Save this FeatureSpace configuration to file.

        Creates a Config object with the default runner and saves it
        to the specified path.

        Args:
            path: Output file path (.yaml, .yml, or .json extension)

        Raises:
            ValueError: If path doesn't have .yaml, .yml, or .json extension

        Example:
            >>> fs = FeatureSpace.from_list([MeanExtractor(), StdExtractor()])
            >>> fs.save_config("pipeline.yaml")
            >>> fs.save_config("pipeline.json")
        """
        from hyppo import io

        if not isinstance(path, io.Path):
            path = io.Path(path)

        config = io.Config(feature_space=self)

        if path.suffix in [".yaml", ".yml"]:
            io.save_config_yaml(config, path)
        elif path.suffix == ".json":
            io.save_config_json(config, path)
        else:
            msg = (
                f"Path must have .yaml, .yml, or .json extension, "
                f"got {path.suffix}"
            )
            raise ValueError(msg)
