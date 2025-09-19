"""
Tests for ProcessRunner.
"""

import pytest
from hyppo.core import FeatureSpace
from hyppo.runner.processes import ProcessRunner
from tests.fixtures.extractors import (
    SimpleExtractor,
    MediumExtractor,
    AdvancedExtractor,
    ComplexExtractor,
)


class TestProcessRunner:
    """Test ProcessRunner."""

    def test_extraction(self, small_hsi):
        """Test basic extraction with ProcessRunner."""
        pipeline_config = {
            "simple": (SimpleExtractor(), {}),
            "medium": (MediumExtractor(), {"simple_input": "simple"}),
        }

        fs = FeatureSpace(pipeline_config)
        runner = ProcessRunner(num_workers=2)

        results = fs.extract(small_hsi, runner)

        assert "simple" in results
        assert "medium" in results

    def test_layer_based_execution(self, small_hsi):
        """Test that extractors in the same layer can execute in parallel."""
        # Create a pipeline where simple1, simple2, simple3 can execute in parallel (layer 0)
        # medium depends on simple1 (layer 1)
        # advanced depends on medium, simple1, simple2 (layer 2)
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
        }

        fs = FeatureSpace(pipeline_config)
        runner = ProcessRunner(num_workers=2, threads_per_worker=1)

        # Verify layer structure
        layers = fs.feature_graph.get_execution_layers()
        assert len(layers) == 3
        assert set(layers[0]) == {"simple1", "simple2", "simple3"}  # Can execute in parallel
        assert layers[1] == ["medium"]
        assert layers[2] == ["advanced"]

        results = runner.resolve(small_hsi, fs)

        # All extractors should have executed successfully
        assert len(results) == 5
        for extractor_name in ["simple1", "simple2", "simple3", "medium", "advanced"]:
            assert extractor_name in results

    def test_complex_pipeline(self, small_hsi):
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
        runner = ProcessRunner(num_workers=3, threads_per_worker=2)

        results = runner.resolve(small_hsi, fs)

        assert len(results) == 6

        # Check that complex extractor was executed
        assert "complex" in results

    def test_single_worker_execution(self, small_hsi):
        """Test execution with a single worker."""
        pipeline_config = {
            "simple": (SimpleExtractor(), {}),
            "medium": (MediumExtractor(), {"simple_input": "simple"}),
        }

        fs = FeatureSpace(pipeline_config)
        runner = ProcessRunner(num_workers=1)

        results = runner.resolve(small_hsi, fs)

        assert len(results) == 2
        assert "simple" in results
        assert "medium" in results

    def test_context_manager(self, small_hsi):
        """Test ProcessRunner as context manager."""
        pipeline_config = {
            "simple": (SimpleExtractor(), {}),
            "medium": (MediumExtractor(), {"simple_input": "simple"}),
        }

        fs = FeatureSpace(pipeline_config)
        
        with ProcessRunner(num_workers=2) as runner:
            results = runner.resolve(small_hsi, fs)

        assert len(results) == 2
        assert "simple" in results
        assert "medium" in results

    def test_custom_memory_limit(self, small_hsi):
        """Test ProcessRunner with custom memory limit."""
        pipeline_config = {
            "simple": (SimpleExtractor(), {}),
        }

        fs = FeatureSpace(pipeline_config)
        runner = ProcessRunner(num_workers=1, memory_limit="1GB")

        results = runner.resolve(small_hsi, fs)

        assert len(results) == 1
        assert "simple" in results

    def test_multiple_threads_per_worker(self, small_hsi):
        """Test ProcessRunner with multiple threads per worker."""
        pipeline_config = {
            "simple1": (SimpleExtractor(), {}),
            "simple2": (SimpleExtractor(), {}),
            "simple3": (SimpleExtractor(), {}),
        }

        fs = FeatureSpace(pipeline_config)
        runner = ProcessRunner(num_workers=2, threads_per_worker=2)

        results = runner.resolve(small_hsi, fs)

        assert len(results) == 3
        for name in ["simple1", "simple2", "simple3"]:
            assert name in results

    def test_invalid_parameters(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid number of workers"):
            ProcessRunner(num_workers=0)

        with pytest.raises(ValueError, match="Invalid number of workers"):
            ProcessRunner(num_workers=-1)

        with pytest.raises(ValueError, match="Invalid threads per worker"):
            ProcessRunner(threads_per_worker=0)

        with pytest.raises(ValueError, match="Invalid threads per worker"):
            ProcessRunner(threads_per_worker=-1)

    def test_cluster_cleanup_manually(self):
        """Test that cluster can be cleaned up manually."""
        runner = ProcessRunner(num_workers=1)
        
        # Setup cluster
        runner._setup_cluster()
        assert runner._cluster is not None
        assert runner._client is not None
        
        # Cleanup cluster
        runner._cleanup_cluster()
        assert runner._cluster is None
        assert runner._client is None

    def test_direct_vs_featurespace_extract(self, small_hsi):
        """Test using ProcessRunner both directly and through FeatureSpace.extract()."""
        pipeline_config = {
            "simple": (SimpleExtractor(), {}),
            "medium": (MediumExtractor(), {"simple_input": "simple"}),
        }

        fs = FeatureSpace(pipeline_config)
        runner = ProcessRunner(num_workers=2)

        # Direct call to runner.resolve()
        results1 = runner.resolve(small_hsi, fs)

        # Call through FeatureSpace.extract()
        results2 = fs.extract(small_hsi, runner)

        # Both should produce the same results
        assert len(results1) == len(results2) == 2
        assert "simple" in results1 and "simple" in results2
        assert "medium" in results1 and "medium" in results2

    def test_dask_graph_generation(self, small_hsi):
        """Test that Dask graph is generated correctly for ProcessRunner."""
        pipeline_config = {
            "simple1": (SimpleExtractor(), {}),
            "simple2": (SimpleExtractor(), {}),
            "medium": (MediumExtractor(), {"simple_input": "simple1"}),
        }

        fs = FeatureSpace(pipeline_config)
        runner = ProcessRunner(num_workers=2)

        # Test graph generation
        dask_graph = runner._build_dask_graph(small_hsi, fs.feature_graph)

        # Verify graph structure
        assert 'hsi_data' in dask_graph
        assert 'simple1' in dask_graph  
        assert 'simple2' in dask_graph
        assert 'medium' in dask_graph

        # Verify HSI data is stored as constant
        assert dask_graph['hsi_data'] is small_hsi

        # Verify task structure for extractors
        simple1_task = dask_graph['simple1']
        assert isinstance(simple1_task, tuple)
        assert len(simple1_task) >= 4  # function, extractor, hsi_data, input_names, defaults

        medium_task = dask_graph['medium']
        assert isinstance(medium_task, tuple)
        # Medium should have dependency on simple1
        assert 'simple1' in medium_task  # simple1 dependency should be in task args

    def test_graph_execution_order_independence(self, small_hsi):
        """Test that graph execution works regardless of definition order."""
        # Define extractors in non-topological order
        pipeline_config = {
            "advanced": (
                AdvancedExtractor(),
                {
                    "medium_input": "medium",
                    "simple_input1": "simple1",
                    "simple_input2": "simple2",
                },
            ),
            "medium": (MediumExtractor(), {"simple_input": "simple1"}),
            "simple1": (SimpleExtractor(), {}),
            "simple2": (SimpleExtractor(), {}),
        }

        fs = FeatureSpace(pipeline_config)
        runner = ProcessRunner(num_workers=2)

        # Should work despite non-topological order in config
        results = runner.resolve(small_hsi, fs)

        assert len(results) == 4
        for name in ["simple1", "simple2", "medium", "advanced"]:
            assert name in results