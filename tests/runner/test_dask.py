"""Tests for Dask-based runners."""

from dask.distributed import Client, LocalCluster
import pytest

from hyppo.core import FeatureCollection, FeatureSpace, HSI
from hyppo.extractor.base import Extractor
from hyppo.runner import (
    DaskProcessesRunner,
    DaskRunner,
    DaskThreadsRunner,
)


class SimpleTestExtractor(Extractor):
    """Simple extractor for testing."""

    def _extract(self, data: HSI, **inputs) -> dict:
        """Extract simple test value."""
        return {"simple_value": 1.0}


class ProducerExtractor(Extractor):
    """Producer whose feature_name collides with a downstream input name."""

    def _extract(self, data: HSI, **inputs) -> dict:
        """Produce a payload consumed by ColludingConsumerExtractor."""
        return {"payload": 42}


class ColludingConsumerExtractor(Extractor):
    """Consumer that names its input exactly like the producer's feature name.

    This collision is what previously triggered a Dask runner crash: Dask
    walked the ``input_names`` list passed to the task and, finding a string
    equal to another graph key, recursively substituted it with that task's
    result. The metadata list then contained a dict, which produced
    ``TypeError: unhashable type: 'dict'`` when used as a mapping key.
    """

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Declare an input whose name matches the producer's feature name."""
        return {
            "producer": {"extractor": ProducerExtractor, "required": True}
        }

    def _extract(self, data: HSI, **inputs) -> dict:
        """Return the consumed payload to assert it was routed correctly."""
        return {"consumed": inputs["producer"]["payload"]}


class DependentTestExtractor(Extractor):
    """Extractor with dependencies for testing."""

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Return test dependencies."""
        return {
            "simple_input": {
                "extractor": SimpleTestExtractor,
                "required": True,
            }
        }

    def _extract(self, data: HSI, **inputs) -> dict:
        """Extract dependent test value."""
        return {"dependent_value": 2.0}


class OptionalDependencyExtractor(Extractor):
    """Extractor with optional dependency and default."""

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Return test dependencies."""
        return {
            "optional_input": {
                "extractor": SimpleTestExtractor,
                "required": False,
            }
        }

    @classmethod
    def get_input_default(cls, input_name: str):
        """Return test default extractor."""
        if input_name == "optional_input":
            return SimpleTestExtractor()
        return None

    def _extract(self, data: HSI, **inputs) -> dict:
        if "optional_input" in inputs:
            return {
                "has_input": True,
                "value": inputs["optional_input"]["simple_value"],
            }
        return {"has_input": False}


class TestDaskRunner:
    """Test cases for DaskRunner class."""

    def test_can_instantiate_with_client(self):
        """Test that DaskRunner can be instantiated with a client."""
        # Arrange: Create cluster and client
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            silence_logs=True,
        )
        client = Client(cluster)

        # Act: Create runner
        runner = DaskRunner(client)

        # Assert: Instance created
        assert isinstance(runner, DaskRunner)
        assert runner._client is client
        assert runner._cluster is cluster

        # Cleanup
        client.close()
        cluster.close()

    def test_threads_runner_init(self):
        """Test DaskThreadsRunner initialization."""
        # Act: Create runner with specific thread count
        runner = DaskThreadsRunner(num_threads=2)

        # Assert: Runner created with cluster attached
        assert isinstance(runner, DaskRunner)
        assert isinstance(runner, DaskThreadsRunner)
        assert runner._client is not None
        assert runner._cluster is not None

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_threads_runner_default_threads(self):
        """Test DaskThreadsRunner with default thread count."""
        # Act: Create runner without specifying threads
        runner = DaskThreadsRunner()

        # Assert: Runner created successfully
        assert isinstance(runner, DaskThreadsRunner)
        assert runner._client is not None
        assert runner._cluster is not None

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_threads_runner_invalid_threads(self):
        """Test DaskThreadsRunner raises error for invalid thread count."""
        # Act & Assert: Invalid thread count
        with pytest.raises(ValueError, match="Invalid number of threads"):
            DaskThreadsRunner(num_threads=0)

        with pytest.raises(ValueError, match="Invalid number of threads"):
            DaskThreadsRunner(num_threads=-1)

    def test_processes_runner_init(self):
        """Test DaskProcessesRunner initialization."""
        # Act: Create runner with specific worker count
        runner = DaskProcessesRunner(num_workers=2, threads_per_worker=1)

        # Assert: Runner created with cluster attached
        assert isinstance(runner, DaskRunner)
        assert isinstance(runner, DaskProcessesRunner)
        assert runner._client is not None
        assert runner._cluster is not None

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_processes_runner_invalid_workers(self):
        """Test DaskProcessesRunner raises error for invalid workers."""
        # Act & Assert: Invalid worker count
        with pytest.raises(ValueError, match="Invalid number of workers"):
            DaskProcessesRunner(num_workers=0)

        with pytest.raises(ValueError, match="Invalid number of workers"):
            DaskProcessesRunner(num_workers=-1)

    def test_processes_runner_invalid_threads_per_worker(self):
        """Test DaskProcessesRunner raises error for invalid threads."""
        # Act & Assert: Invalid threads per worker
        with pytest.raises(ValueError, match="Invalid threads per worker"):
            DaskProcessesRunner(threads_per_worker=0)

        with pytest.raises(ValueError, match="Invalid threads per worker"):
            DaskProcessesRunner(threads_per_worker=-1)

    def test_resolve_single_extractor(self, small_hsi):
        """Test resolving single extractor without dependencies."""
        # Arrange: Create runner and feature space
        runner = DaskThreadsRunner(num_threads=1)
        extractor = SimpleTestExtractor()
        fs = FeatureSpace.from_list([extractor])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Verify results
        assert isinstance(results, FeatureCollection)
        assert len(results) == 1
        assert "simple_test" in results
        assert "simple_value" in results["simple_test"]["data"]

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_resolve_with_dependencies(self, small_hsi):
        """Test resolving extractors with dependencies."""
        # Arrange: Create runner with dependent extractors
        runner = DaskThreadsRunner(num_threads=2)
        simple = SimpleTestExtractor()
        dependent = DependentTestExtractor()
        fs = FeatureSpace.from_list([simple, dependent])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Verify both extractors executed
        assert len(results) == 2
        assert "simple_test" in results
        assert "dependent_test" in results
        assert "dependent_value" in results["dependent_test"]["data"]

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_resolve_complex_pipeline(self, small_hsi):
        """Test complex pipeline with multiple extractors."""
        # Arrange: Create pipeline with test extractors
        from tests.fixtures.extractors import (
            AdvancedExtractor,
            MediumExtractor,
            SimpleExtractor,
        )

        runner = DaskThreadsRunner(num_threads=2)
        fs = FeatureSpace.from_list(
            [SimpleExtractor(), MediumExtractor(), AdvancedExtractor()]
        )

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: All extractors produced results
        assert len(results) == 3
        assert "simple" in results
        assert "medium" in results
        assert "advanced" in results

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_build_dask_graph(self, small_hsi):
        """Test _build_dask_graph creates correct graph structure."""
        # Arrange: Create runner and feature space
        runner = DaskThreadsRunner(num_threads=1)
        simple = SimpleTestExtractor()
        dependent = DependentTestExtractor()
        fs = FeatureSpace.from_list([simple, dependent])

        # Act: Build graph
        graph = runner._build_dask_graph(small_hsi, fs.feature_graph)

        # Assert: Verify graph structure
        assert "hsi_data" in graph
        assert graph["hsi_data"] is small_hsi
        assert "simple_test" in graph
        assert "dependent_test" in graph

        # Verify task structure
        simple_task = graph["simple_test"]
        assert isinstance(simple_task, tuple)
        assert len(simple_task) >= 4

        dependent_task = graph["dependent_test"]
        assert isinstance(dependent_task, tuple)
        assert "simple_test" in dependent_task

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_resolve_empty_feature_space(self, small_hsi):
        """Test resolving with empty feature space."""
        # Arrange: Create empty feature space
        runner = DaskThreadsRunner(num_threads=1)
        fs = FeatureSpace({})

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Empty results
        assert len(results) == 0

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_graph_execution_order_independence(self, small_hsi):
        """Test that graph execution works regardless of definition order."""
        # Arrange: Define extractors in non-topological order
        from tests.fixtures.extractors import MediumExtractor, SimpleExtractor

        runner = DaskThreadsRunner(num_threads=2)
        # Add in reverse dependency order (Medium depends on Simple)
        fs = FeatureSpace.from_list([MediumExtractor(), SimpleExtractor()])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Both executed successfully
        assert len(results) == 2
        assert "simple" in results
        assert "medium" in results

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_integration_with_feature_space_extract(self, small_hsi):
        """Test using DaskRunner through FeatureSpace.extract()."""
        # Arrange: Create runner and feature space
        from tests.fixtures.extractors import MediumExtractor, SimpleExtractor

        runner = DaskThreadsRunner(num_threads=2)
        fs = FeatureSpace.from_list([SimpleExtractor(), MediumExtractor()])

        # Act: Call through FeatureSpace.extract()
        results = fs.extract(small_hsi, runner)

        # Assert: Results produced
        assert len(results) == 2
        assert "simple" in results
        assert "medium" in results

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_optional_dependency_with_default(self, small_hsi):
        """Test extractor with optional dependency using default."""
        # Arrange: Create runner with only optional dependency extractor
        runner = DaskThreadsRunner(num_threads=1)
        optional = OptionalDependencyExtractor()
        fs = FeatureSpace.from_list([optional])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Default was used
        assert len(results) == 1
        assert "optional_dependency" in results
        assert results["optional_dependency"]["data"]["has_input"] is True

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_optional_dependency_provided(self, small_hsi):
        """Test optional dependency when source is provided."""
        # Arrange: Create runner with optional dependency and source
        runner = DaskThreadsRunner(num_threads=2)
        simple = SimpleTestExtractor()
        optional = OptionalDependencyExtractor()
        fs = FeatureSpace.from_list([simple, optional])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Optional input was used
        assert len(results) == 2
        assert results["optional_dependency"]["data"]["has_input"] is True
        assert "value" in results["optional_dependency"]["data"]

        # Cleanup
        runner._client.close()
        runner._cluster.close()

    def test_resolve_with_input_name_colliding_with_feature_name(
        self, small_hsi
    ):
        """Dask must not recursively resolve ``input_names`` as graph keys.

        Regression test: when an extractor declares an input whose name
        matches another extractor's ``feature_name()``, Dask previously
        walked the ``input_names`` metadata list and substituted the
        matching string with the upstream task's result dict. That turned
        the dict into a mapping key and raised ``TypeError: unhashable
        type: 'dict'``. The metadata list must be treated as a literal.
        """
        # Arrange: producer's feature_name ("producer") equals the input
        # name declared by ColludingConsumerExtractor.
        runner = DaskThreadsRunner(num_threads=2)
        fs = FeatureSpace.from_list(
            [ProducerExtractor(), ColludingConsumerExtractor()]
        )
        assert "producer" in fs.feature_graph.extractors
        consumer_mapping = fs.feature_graph.get_input_mapping_for(
            "colluding_consumer"
        )
        assert consumer_mapping == {"producer": "producer"}

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Consumer received the producer's payload intact.
        assert "colluding_consumer" in results
        assert results["colluding_consumer"]["data"]["consumed"] == 42

        # Cleanup
        runner._client.close()
        runner._cluster.close()
