import networkx as nx

from hyppo.extractor.base import Extractor


class FeatureDependencyGraph:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.extractors = {}
        self.input_mappings = {}

    def add_extractor(
        self, name: str, extractor: Extractor, input_mapping: dict | None = None
    ):
        """
        Add an extractor with its input mapping to the dependency graph.

        Args:
            name: Unique name for this extractor instance
            extractor: The extractor instance
            input_mapping: dict mapping {input_name: source_extractor_name}
                          If None, no inputs are mapped (extractor has no dependencies)
        """
        if input_mapping is None:
            input_mapping = {}

        # Add node to graph
        self.graph.add_node(name, extractor=extractor)
        self.extractors[name] = extractor
        self.input_mappings[name] = input_mapping

        # Add edges for dependencies
        for input_name, source_name in input_mapping.items():
            # Add edge from source to this extractor
            self.graph.add_edge(source_name, name, input_name=input_name)

    def validate(self) -> None:
        """
        Validate the dependency graph for cycles and type compatibility.

        Raises:
            ValueError: If circular dependencies are detected
            TypeError: If input type mismatches are found
            ValueError: If required inputs are missing
        """
        # Check for cycles using NetworkX
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Circular dependencies detected: {cycles}")

        # Validate types and requirements
        self._validate_types_and_requirements()

    def _validate_types_and_requirements(self) -> None:
        """Validate input types and requirements for all extractors."""
        for node_name in self.graph.nodes():
            extractor = self.extractors[node_name]
            input_deps = extractor.get_input_dependencies()
            input_mapping = self.input_mappings.get(node_name, {})

            # Check each declared input dependency
            for input_name, dep_spec in input_deps.items():
                # Check if required input is missing
                if dep_spec.get("required", True) and input_name not in input_mapping:
                    raise ValueError(
                        f"Extractor '{node_name}' requires input '{input_name}' but it was not provided"
                    )

                # If input is mapped, validate the source extractor type
                if input_name in input_mapping:
                    source_name = input_mapping[input_name]
                    if source_name not in self.extractors:
                        raise ValueError(
                            f"Source extractor '{source_name}' not found for input '{input_name}' of '{node_name}'"
                        )

                    source_extractor = self.extractors[source_name]
                    if not isinstance(source_extractor, dep_spec["extractor"]):
                        raise TypeError(
                            f"Type mismatch for input '{input_name}' of '{node_name}': "
                            f"expected {dep_spec['extractor'].__name__}, "
                            f"got {type(source_extractor).__name__}"
                        )

    def get_execution_order(self) -> list[str]:
        """
        Get the execution order for the provided dependencies.

        Returns:
            list of extractor names in execution order
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            raise ValueError(f"Cannot determine execution order: {e}")

    def get_execution_layers(self) -> list[list[str]]:
        """
        Get extractors organized in execution layers for parallel execution.

        Each layer has dependencies on the previous layer. Except for the first layer.

        Returns:
            list of layers, where each layer contains extractors that can run in parallel
        """
        layers = []
        remaining_nodes = set(self.graph.nodes())

        while remaining_nodes:
            # Find nodes with no incoming edges from remaining nodes
            current_layer = []
            for node in list(remaining_nodes):
                # Check if all predecessors are already processed
                predecessors = set(self.graph.predecessors(node))
                if predecessors.isdisjoint(remaining_nodes):
                    current_layer.append(node)

            if not current_layer:
                # This should not happen if graph is acyclic
                raise ValueError(
                    f"Cannot compute execution layers. Remaining nodes: {remaining_nodes}"
                )

            layers.append(current_layer)
            remaining_nodes.difference_update(current_layer)

        return layers

    def get_dependencies_for(self, extractor_name: str) -> set[str]:
        """Get all transitive dependencies for an extractor."""
        if extractor_name not in self.graph:
            return set()
        return set(nx.ancestors(self.graph, extractor_name))

    def get_dependents_of(self, extractor_name: str) -> set[str]:
        """Get all extractors that depend on this one."""
        if extractor_name not in self.graph:
            return set()
        return set(nx.descendants(self.graph, extractor_name))

    def get_input_mapping_for(self, extractor_name: str) -> dict[str, str]:
        """Get the input mapping for a specific extractor."""
        return self.input_mappings.get(extractor_name, {}).copy()
