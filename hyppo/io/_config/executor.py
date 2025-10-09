from .models import Config, ExtractorConfig
from .validator import ConfigValidator
from hyppo.core import FeatureSpace
from hyppo.extractor.base import Extractor
from hyppo.extractor import registry



class ConfigExecutor:
    """Executor that builds FeatureSpace from configuration."""

    def __init__(self, config: Config, validate: bool = True):
        """
        Initialize executor with configuration.

        Args:
            config: Configuration object to execute
            validate: Whether to validate configuration before execution
        """
        self.config = config

        if validate:
            validator = ConfigValidator()
            validator.validate(config)

    def build_feature_space(self) -> FeatureSpace:
        """
        Build FeatureSpace from configuration.

        Returns:
            Configured FeatureSpace ready for extraction
        """
        extractor_configs = {}

        # Process each extractor in pipeline
        for name, extractor_config in self.config.pipeline.extractors.items():
            # Create extractor instance
            extractor = self._create_extractor(extractor_config)

            # Map inputs to source extractor names
            input_mapping = extractor_config.inputs

            # Store configuration tuple
            extractor_configs[name] = (extractor, input_mapping)

        return FeatureSpace(extractor_configs)

    def _create_extractor(self, extractor_config: ExtractorConfig) -> Extractor:
        """
        Create extractor instance from configuration.

        Args:
            extractor_config: Extractor configuration

        Returns:
            Instantiated extractor
        """
        extractor_type = extractor_config.extractor_type
        extractor_params = extractor_config.extractor_params

        if not registry.is_registered(extractor_type):
            raise ValueError(f"Unknown extractor type: {extractor_type}")

        extractor_class = registry.get(extractor_type)

        try:
            return extractor_class(**extractor_params)
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate {extractor_type} with parameters "
                f"{extractor_params}: {e}"
            )

        except ImportError as e:
            raise RuntimeError(f"Failed to import extractors: {e}")
