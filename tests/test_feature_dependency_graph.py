"""
Tests for FeatureDependencyGraph.
"""

import pytest
import numpy as np
from hyppo.extractor.base import Extractor, InputDependency
from hyppo.extractor.mean import MeanExtractor
from hyppo.extractor.std import StdExtractor
from hyppo.feature_dependency_graph import FeatureDependencyGraph
from tests.fixtures.extractors import (
    SimpleExtractor,
    MediumExtractor,
    AdvancedExtractor,
)


class TestFeatureDependencyGraph:
    """Test the FeatureDependencyGraph class."""

    def test_empty_graph(self):
        """Test empty feature dependency graph."""
        graph = FeatureDependencyGraph()

        assert len(graph.extractors) == 0
        assert graph.graph.number_of_nodes() == 0
        assert graph.graph.number_of_edges() == 0

    def test_add_extractor_no_dependencies(self):
        """Test adding extractor with no dependencies."""
        graph = FeatureDependencyGraph()
        extractor = SimpleExtractor()

        graph.add_extractor("simple", extractor, {})
        graph.validate()

        assert "simple" in graph.extractors
        assert graph.graph.number_of_nodes() == 1
        assert graph.graph.number_of_edges() == 0

    def test_add_extractor_with_dependencies(self):
        """Test adding extractor with typed dependencies."""
        graph = FeatureDependencyGraph()

        # Add mean extractor (no dependencies)
        simple = SimpleExtractor()
        graph.add_extractor("simple", simple, {})

        # Add medium extractor with simple dependency
        medium = MediumExtractor()
        graph.add_extractor("medium", medium, {"simple_input": "simple"})

        graph.validate()

        assert len(graph.extractors) == 2
        assert graph.graph.number_of_edges() == 1
        assert graph.graph.has_edge("simple", "medium")

    def test_validation_type_mismatch(self):
        """Test validation catches type mismatches."""
        graph = FeatureDependencyGraph()

        # Add wrong type of extractor
        std = StdExtractor()
        graph.add_extractor("mean", std, {})  # std as mean

        # Add medium that expects SimpleExtractor but gets StdExtractor
        medium = MediumExtractor()
        graph.add_extractor("medium", medium, {"simple_input": "mean"})

        with pytest.raises(TypeError, match="Type mismatch"):
            graph.validate()

    def test_validation_missing_required_input(self):
        """Test validation catches missing required inputs."""
        graph = FeatureDependencyGraph()

        # Use AdvancedExtractor which has required inputs
        advanced = AdvancedExtractor()
        graph.add_extractor("advanced", advanced, {})

        with pytest.raises(ValueError, match="requires input"):
            graph.validate()

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        graph = FeatureDependencyGraph()

        # Create two extractors that depend on each other
        class CircularA(Extractor):
            input_dependencies = {"b": InputDependency("b", Extractor, required=True)}

            def extract(self, data, **inputs):
                return {"features": np.ones((data.height, data.width))}

        class CircularB(Extractor):
            input_dependencies = {"a": InputDependency("a", Extractor, required=True)}

            def extract(self, data, **inputs):
                return {"features": np.ones((data.height, data.width))}

        a = CircularA()
        b = CircularB()

        graph.add_extractor("a", a, {"b": "b"})
        graph.add_extractor("b", b, {"a": "a"})

        with pytest.raises(ValueError, match="Circular dependencies"):
            graph.validate()

    def test_dependencies_execution_order(self):
        """Test topological sorting."""
        graph = FeatureDependencyGraph()

        # Create chain: simple -> medium -> advanced
        simple = SimpleExtractor()
        simple2 = SimpleExtractor()
        medium = MediumExtractor()
        advanced = AdvancedExtractor()

        graph.add_extractor("simple", simple, {})
        graph.add_extractor("simple2", simple2, {})
        graph.add_extractor("medium", medium, {"simple_input": "simple"})
        graph.add_extractor(
            "advanced",
            advanced,
            {
                "medium_input": "medium",
                "simple_input1": "simple",
                "simple_input2": "simple2",
            },
        )

        graph.validate()

        execution_order = graph.get_execution_order()

        # Check that dependencies come before dependents
        simple_idx = execution_order.index("simple")
        simple2_idx = execution_order.index("simple2")
        medium_idx = execution_order.index("medium")
        advanced_idx = execution_order.index("advanced")

        assert simple_idx < medium_idx
        assert simple_idx < advanced_idx
        assert simple2_idx < advanced_idx
        assert medium_idx < advanced_idx

        execution_layers = graph.get_execution_layers()
        assert len(execution_layers) == 3
        assert set(execution_layers[0]) == {"simple", "simple2"}
        assert execution_layers[1] == ["medium"]
        assert execution_layers[2] == ["advanced"]
