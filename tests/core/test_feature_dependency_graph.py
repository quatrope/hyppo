"""Tests for FeatureDependencyGraph."""

import numpy as np
import pytest

from hyppo.core import FeatureDependencyGraph
from hyppo.extractor.base import Extractor
from hyppo.extractor.savi import SAVIExtractor
from tests.fixtures.extractors import (
    AdvancedExtractor,
    MediumExtractor,
    SimpleExtractor,
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
        std = SAVIExtractor()
        graph.add_extractor("mean", std, {})  # std as mean

        # Add medium that expects SimpleExtractor but gets SAVIExtractor
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
            @classmethod
            def get_input_dependencies(cls) -> dict:
                return {"b": {"extractor": Extractor, "required": True}}

            def _extract(self, data, **inputs):
                return {"features": np.ones((data.height, data.width))}

        class CircularB(Extractor):
            @classmethod
            def get_input_dependencies(cls) -> dict:
                return {"a": {"extractor": Extractor, "required": True}}

            def _extract(self, data, **inputs):
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

    def test_add_extractor_with_none_input_mapping(self):
        """Test adding extractor with input_mapping=None."""
        graph = FeatureDependencyGraph()
        extractor = SimpleExtractor()

        graph.add_extractor("simple", extractor, None)
        graph.validate()

        assert "simple" in graph.extractors
        assert graph.input_mappings["simple"] == {}

    def test_validation_source_extractor_not_found(self):
        """Test validation when source extractor doesn't exist."""
        graph = FeatureDependencyGraph()

        medium = MediumExtractor()
        graph.add_extractor("medium", medium, {"simple_input": "nonexistent"})

        with pytest.raises(ValueError, match="Source extractor .* not found"):
            graph.validate()

    def test_get_dependencies_for_nonexistent(self):
        """Test get_dependencies_for with nonexistent extractor."""
        graph = FeatureDependencyGraph()
        simple = SimpleExtractor()
        graph.add_extractor("simple", simple, {})

        deps = graph.get_dependencies_for("nonexistent")
        assert deps == set()

    def test_get_dependencies_for_existing(self):
        """Test get_dependencies_for with existing extractor."""
        graph = FeatureDependencyGraph()
        simple = SimpleExtractor()
        medium = MediumExtractor()

        graph.add_extractor("simple", simple, {})
        graph.add_extractor("medium", medium, {"simple_input": "simple"})

        deps = graph.get_dependencies_for("medium")
        assert deps == {"simple"}

    def test_get_dependents_of_nonexistent(self):
        """Test get_dependents_of with nonexistent extractor."""
        graph = FeatureDependencyGraph()
        simple = SimpleExtractor()
        graph.add_extractor("simple", simple, {})

        dependents = graph.get_dependents_of("nonexistent")
        assert dependents == set()

    def test_get_dependents_of_existing(self):
        """Test get_dependents_of with existing extractor."""
        graph = FeatureDependencyGraph()
        simple = SimpleExtractor()
        medium = MediumExtractor()

        graph.add_extractor("simple", simple, {})
        graph.add_extractor("medium", medium, {"simple_input": "simple"})

        dependents = graph.get_dependents_of("simple")
        assert dependents == {"medium"}

    def test_get_input_mapping_for(self):
        """Test get_input_mapping_for method."""
        graph = FeatureDependencyGraph()
        simple = SimpleExtractor()
        medium = MediumExtractor()

        graph.add_extractor("simple", simple, {})
        graph.add_extractor("medium", medium, {"simple_input": "simple"})

        mapping = graph.get_input_mapping_for("medium")
        assert mapping == {"simple_input": "simple"}

        empty_mapping = graph.get_input_mapping_for("simple")
        assert empty_mapping == {}

    def test_get_execution_order_with_cycle(self):
        """Test get_execution_order with cyclic graph raises ValueError."""
        graph = FeatureDependencyGraph()

        class CircularA(Extractor):
            @classmethod
            def get_input_dependencies(cls) -> dict:
                return {}

            def _extract(self, data, **inputs):
                return {"features": np.ones((data.height, data.width))}

        class CircularB(Extractor):
            @classmethod
            def get_input_dependencies(cls) -> dict:
                return {}

            def _extract(self, data, **inputs):
                return {"features": np.ones((data.height, data.width))}

        a = CircularA()
        b = CircularB()

        graph.add_extractor("a", a, {"b": "b"})
        graph.add_extractor("b", b, {"a": "a"})

        with pytest.raises(
            ValueError, match="Cannot determine execution order"
        ):
            graph.get_execution_order()
