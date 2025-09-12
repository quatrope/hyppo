import pytest
from pathlib import Path

from hyppo.io import ExtractorConfig, PipelineConfig, Config


class TestExtractorConfig:
    """Test ExtractorConfig functionality."""

    def test_simple_extractor(self):
        """Test simple extractor configuration."""
        config = ExtractorConfig(extractor="MeanExtractor")

        assert config.extractor_type == "MeanExtractor"
        assert config.extractor_params == {}
        assert config.inputs == {}

    def test_extractor_with_params(self):
        """Test extractor with parameters."""
        extractor_spec = {
            "type": "CustomExtractor",
            "param1": "value1",
            "param2": 42,
            "param3": True,
        }
        config = ExtractorConfig(extractor=extractor_spec)

        assert config.extractor_type == "CustomExtractor"
        assert config.extractor_params == {
            "param1": "value1",
            "param2": 42,
            "param3": True,
        }

    def test_extractor_with_inputs(self):
        """Test extractor with input dependencies."""
        inputs = {"mean_data": "mean", "std_data": "std"}
        config = ExtractorConfig(extractor="CustomExtractor", inputs=inputs)

        assert config.extractor_type == "CustomExtractor"
        assert config.inputs == inputs

    def test_invalid_extractor_spec(self):
        """Test invalid extractor specification."""
        invalid_spec = {"param": "value"}  # Missing 'type'
        config = ExtractorConfig(extractor=invalid_spec)

        with pytest.raises(ValueError, match="Invalid extractor specification"):
            _ = config.extractor_type


class TestPipelineConfig:
    """Test PipelineConfig functionality."""

    def test_empty_pipeline(self):
        """Test empty pipeline configuration."""
        pipeline = PipelineConfig()

        assert len(pipeline.extractors) == 0
        assert pipeline.get_extractor_names() == []

    def test_add_extractor(self):
        """Test adding extractors to pipeline."""
        pipeline = PipelineConfig()

        extractor1 = ExtractorConfig(extractor="MeanExtractor")
        extractor2 = ExtractorConfig(extractor="StdExtractor")

        pipeline.add_extractor("mean", extractor1)
        pipeline.add_extractor("std", extractor2)

        assert len(pipeline.extractors) == 2
        assert "mean" in pipeline.extractors
        assert "std" in pipeline.extractors
        assert set(pipeline.get_extractor_names()) == {"mean", "std"}

    def test_get_extractor(self):
        """Test getting extractor from pipeline."""
        pipeline = PipelineConfig()
        extractor = ExtractorConfig(extractor="MeanExtractor")
        pipeline.add_extractor("mean", extractor)

        retrieved = pipeline.get_extractor("mean")
        assert retrieved is extractor

        missing = pipeline.get_extractor("nonexistent")
        assert missing is None
