import attrs
from typing import Dict, Any, List, Optional
from pathlib import Path


@attrs.define
class ExtractorConfig:
    """Configuration for a single extractor in the pipeline."""

    extractor: str | Dict[str, Any]
    inputs: Dict[str, str] = attrs.field(factory=dict)

    @property
    def extractor_type(self) -> str:
        """Get the extractor type name."""
        if isinstance(self.extractor, str):
            return self.extractor
        elif isinstance(self.extractor, dict) and "type" in self.extractor:
            return self.extractor["type"]
        else:
            raise ValueError(f"Invalid extractor specification: {self.extractor}")

    @property
    def extractor_params(self) -> Dict[str, Any]:
        """Get extractor parameters (excluding 'type')."""
        if isinstance(self.extractor, dict):
            return {k: v for k, v in self.extractor.items() if k != "type"}
        return {}


@attrs.define
class PipelineConfig:
    """Configuration for the complete extraction pipeline."""

    extractors: Dict[str, ExtractorConfig] = attrs.field(factory=dict)

    def add_extractor(self, name: str, extractor_config: ExtractorConfig) -> None:
        """Add an extractor to the pipeline."""
        self.extractors[name] = extractor_config

    def get_extractor(self, name: str) -> Optional[ExtractorConfig]:
        """Get an extractor configuration by name."""
        return self.extractors.get(name)

    def get_extractor_names(self) -> List[str]:
        """Get all extractor names in the pipeline."""
        return list(self.extractors.keys())


@attrs.define
class Config:
    """Complete configuration for HYPPO feature extraction."""

    pipeline: PipelineConfig
