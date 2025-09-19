from typing import Dict, Sequence, Tuple, List, Union, TYPE_CHECKING
from pathlib import Path

from ._feature_dependency_graph import FeatureDependencyGraph
from ._hsi import HSI

if TYPE_CHECKING:
    from hyppo.extractor.base import Extractor


class FeatureSpace:

    def __init__(
        self,
        extractor_configs
    ):
        """
        Initialize FeatureSpace with feature extractor configurations.

        Args:
            extractor_configs: Either:
                - Dict of {name: (extractor, input_mapping)} where input_mapping
                  is {input_name: source_extractor_name}
                - List of Extractor instances (shortcut API with automatic dependency resolution)
        """
        # Handle list input by converting to proper config dict
        if isinstance(extractor_configs, list):
            extractor_configs = self._convert_list_to_config(extractor_configs)

        self.extractor_configs = extractor_configs
        self.extractors = {
            name: config[0] for name, config in extractor_configs.items()
        }
        self.feature_graph = self._build_feature_dependency_graph()

    def _convert_list_to_config(self, extractors):
        """
        Convert list of extractors to configuration dict with automatic dependency resolution.
        """
        if not extractors:
            return {}

        # Create name-to-extractor mapping and check for duplicates
        extractor_mapping = {}
        type_to_name = {}

        for extractor in extractors:
            extractor_name = extractor.feature_name()
            extractor_type = type(extractor)

            # Check for duplicate names
            if extractor_name in extractor_mapping:
                raise ValueError(
                    f"Duplicate extractor name '{extractor_name}'. "
                    f"Cannot have multiple extractors of the same type in list."
                )

            # Check for duplicate types (which would lead to same name)
            if extractor_type in type_to_name:
                raise ValueError(
                    f"Duplicate extractor type '{extractor_type.__name__}' found. "
                    f"Both would generate the same name '{extractor_name}'. "
                    f"Cannot disambiguate dependencies."
                )

            extractor_mapping[extractor_name] = extractor
            type_to_name[extractor_type] = extractor_name

        # Resolve dependencies for each extractor
        config_dict = {}

        for extractor_name, extractor in extractor_mapping.items():
            input_dependencies = extractor.get_input_dependencies()
            input_mapping = {}

            # Resolve each declared dependency
            for input_name, dep_spec in input_dependencies.items():
                required_type = dep_spec.extractor_type

                # Look for an extractor of the required type
                matching_extractor_name = None
                for candidate_name, candidate_extractor in extractor_mapping.items():
                    if isinstance(candidate_extractor, required_type):
                        if matching_extractor_name is not None:
                            # Multiple matches - ambiguous
                            raise ValueError(
                                f"Ambiguous dependency: extractor '{extractor_name}' requires "
                                f"input '{input_name}' of type '{required_type.__name__}', "
                                f"but found multiple candidates: '{matching_extractor_name}' and '{candidate_name}'"
                            )
                        matching_extractor_name = candidate_name

                if matching_extractor_name is not None:
                    input_mapping[input_name] = matching_extractor_name
                elif dep_spec.required:
                    raise ValueError(
                        f"Missing required dependency: extractor '{extractor_name}' requires "
                        f"input '{input_name}' of type '{required_type.__name__}', "
                        f"but no such extractor found in the list"
                    )

            config_dict[extractor_name] = (extractor, input_mapping)

        return config_dict

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

    @classmethod
    def from_config(
        cls, config_path: str | Path, validate: bool = True
    ) -> "FeatureSpace":
        """
        Create FeatureSpace from configuration file.

        Args:
            config_path: Path to configuration file (JSON or YAML)
            validate: Whether to validate configuration before building

        Returns:
            Configured FeatureSpace ready for extraction

        Example:
            >>> fs = FeatureSpace.from_config("config.yaml")
            >>> hsi = hyppo.io.load_h5("data.h5")
            >>> results = fs.extract(hsi)
        """
        from hyppo.io import parse_config, ConfigExecutor

        config = parse_config(config_path)
        executor = ConfigExecutor(config, validate=validate)
        return executor.build_feature_space()

    @classmethod
    def from_list(
        cls, extractors: "List[Extractor]"
    ):
        """
        Convert list of extractors to configuration dict with automatic dependency resolution.

        Args:
            extractors: List of extractor instances

        Returns:
            Dict configuration with automatic dependency mappings

        Raises:
            ValueError: If duplicate extractor types are found or required dependencies are missing
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
                raise ValueError(
                    f"Duplicate extractor name '{extractor_name}'. "
                    f"Cannot have multiple extractors of the same type in list."
                )

            # Check for duplicate types (which would lead to same name)
            if extractor_type in type_to_name:
                raise ValueError(
                    f"Duplicate extractor type '{extractor_type.__name__}' found. "
                    f"Both would generate the same name '{extractor_name}'. "
                    f"Cannot disambiguate dependencies."
                )

            extractor_mapping[extractor_name] = extractor
            type_to_name[extractor_type] = extractor_name

        # Resolve dependencies for each extractor
        config_dict = {}

        for extractor_name, extractor in extractor_mapping.items():
            input_dependencies = extractor.get_input_dependencies()
            input_mapping = {}

            # Resolve each declared dependency
            for input_name, dep_spec in input_dependencies.items():
                required_type = dep_spec.extractor_type

                # Look for an extractor of the required type
                matching_extractor_name = None
                for candidate_name, candidate_extractor in extractor_mapping.items():
                    if isinstance(candidate_extractor, required_type):
                        if matching_extractor_name is not None:
                            # Multiple matches - ambiguous
                            raise ValueError(
                                f"Ambiguous dependency: extractor '{extractor_name}' requires "
                                f"input '{input_name}' of type '{required_type.__name__}', "
                                f"but found multiple candidates: '{matching_extractor_name}' and '{candidate_name}'"
                            )
                        matching_extractor_name = candidate_name

                if matching_extractor_name is not None:
                    input_mapping[input_name] = matching_extractor_name
                elif dep_spec.required:
                    raise ValueError(
                        f"Missing required dependency: extractor '{extractor_name}' requires "
                        f"input '{input_name}' of type '{required_type.__name__}', "
                        f"but no such extractor found in the list"
                    )

            config_dict[extractor_name] = (extractor, input_mapping)

        
        return cls(config_dict)
