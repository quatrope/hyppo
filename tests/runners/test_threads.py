"""
Tests for ThreadsRunner.
"""

import pytest
from hyppo.core import FeatureSpace
from hyppo.runner.threads import ThreadsRunner
from tests.fixtures.extractors import (
    SimpleExtractor,
    MediumExtractor,
    AdvancedExtractor,
    ComplexExtractor,
)


class TestThreadsRunner:
    """Test ThreadsRunner."""

    def test_extraction(self, small_hsi):
        """Test extraction."""
        pipeline_config = {
            "simple": (SimpleExtractor(), {}),
            "medium": (MediumExtractor(), {"simple_input": "simple"}),
        }

        fs = FeatureSpace(pipeline_config)
        runner = ThreadsRunner(num_workers=1)

        results = fs.extract(small_hsi, runner)

        assert "simple" in results
        assert "medium" in results

    def test_extraction_with_defaults(self, small_hsi):
        """Test extraction using default values for optional inputs."""
        pipeline_config = {
            "simple": (SimpleExtractor(), {}),
            "medium": (
                MediumExtractor(),
                {
                    "simple_input": "simple",  # correct mapping
                },
            ),
            "advanced": (
                AdvancedExtractor(),
                {
                    "medium_input": "simple",  # wrong type: should be MediumExtractor
                    "simple_input1": "simple",
                    "simple_input2": "simple",
                },
            ),
        }

        # This should fail type validation
        with pytest.raises(TypeError):
            FeatureSpace(pipeline_config)

    def test_extraction_complex_pipeline(self, small_hsi):
        """Test complex pipeline with multiple dependencies."""
        pipeline_config = {
            "simple1": (SimpleExtractor(), {}),
            "simple2": (SimpleExtractor(), {}),
            "simple3": (SimpleExtractor(), {}),
            "medium": (MediumExtractor(), {"simple_input": "simple1"}),
            "advanced": (
                AdvancedExtractor(),
                {
                    "medium_input": "medium",
                    "simple_input1": "simple1",
                    "simple_input2": "simple2",
                },
            ),
            "complex": (
                ComplexExtractor(),
                {
                    "simple_input1": "simple1",
                    "medium_input": "medium",
                    "advanced_input1": "advanced",
                    "advanced_input2": "advanced",
                },
            ),
        }

        fs = FeatureSpace(pipeline_config)
        runner = ThreadsRunner(num_workers=2)

        results = fs.extract(small_hsi, runner)

        assert len(results) == 6

        # Check that complex extractor was executed
        assert "complex" in results
