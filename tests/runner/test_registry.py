"""Tests for runner registry system."""

import pytest

from hyppo.runner import (
    BaseRunner,
    DaskProcessesRunner,
    DaskSLURMRunner,
    DaskThreadsRunner,
    LocalProcessRunner,
    SequentialRunner,
    registry,
)



# Check if dask-jobqueue is available for SLURM tests
try:
    from dask_jobqueue import SLURMCluster
    HAS_DASK_JOBQUEUE = True
except ImportError:
    HAS_DASK_JOBQUEUE = False


class TestRunnerRegistry:
    """Tests for runner registry structure and contents."""

    def test_registry_exists(self):
        """Registry should exist."""
        assert registry is not None
        assert len(registry) > 0

    def test_registry_contains_all_runners(self):
        """Registry should contain all expected runner types."""
        expected_runners = {
            "sequential",
            "local",
            "dask-threads",
            "dask-processes",
            "dask-slurm",
        }
        assert set(registry.list_runners()) == expected_runners


class TestCreateRunner:
    """Tests for create_runner factory function."""

    def test_create_sequential_runner(self):
        """Should create SequentialRunner without parameters."""
        runner = registry.get("sequential")
        assert isinstance(runner, SequentialRunner)
        assert isinstance(runner, BaseRunner)

    def test_create_sequential_runner_with_empty_params(self):
        """Should create SequentialRunner with empty params dict."""
        runner = registry.get("sequential", {})
        assert isinstance(runner, SequentialRunner)

    def test_create_local_runner_default(self):
        """Should create LocalProcessRunner with defaults."""
        runner = registry.get("local")
        assert isinstance(runner, LocalProcessRunner)
        assert isinstance(runner, BaseRunner)

    def test_create_local_runner_with_workers(self):
        """Should create LocalProcessRunner with specified workers."""
        runner = registry.get("local", {"num_workers": 4})
        assert isinstance(runner, LocalProcessRunner)

    def test_create_dask_threads_default(self):
        """Should create DaskThreadsRunner with defaults."""
        runner = registry.get("dask-threads")
        assert isinstance(runner, DaskThreadsRunner)
        assert isinstance(runner, BaseRunner)

    def test_create_dask_threads_with_params(self):
        """Should create DaskThreadsRunner with specified threads."""
        runner = registry.get("dask-threads", {"num_threads": 8})
        assert isinstance(runner, DaskThreadsRunner)

    def test_create_dask_processes_default(self):
        """Should create DaskProcessesRunner with defaults."""
        runner = registry.get("dask-processes")
        assert isinstance(runner, DaskProcessesRunner)
        assert isinstance(runner, BaseRunner)

    def test_create_dask_processes_with_workers(self):
        """Should create DaskProcessesRunner with workers."""
        runner = registry.get("dask-processes", {"num_workers": 4})
        assert isinstance(runner, DaskProcessesRunner)

    def test_create_dask_processes_with_threads_per_worker(self):
        """Should create DaskProcessesRunner with threads per worker."""
        runner = registry.get(
            "dask-processes",
            {"num_workers": 4, "threads_per_worker": 2}
        )
        assert isinstance(runner, DaskProcessesRunner)

    def test_create_dask_processes_with_memory_limit(self):
        """Should create DaskProcessesRunner with memory limit."""
        runner = registry.get(
            "dask-processes",
            {"memory_limit": "8GB"}
        )
        assert isinstance(runner, DaskProcessesRunner)

    @pytest.mark.skipif(not HAS_DASK_JOBQUEUE, reason="dask-jobqueue not installed")
    def test_create_dask_slurm_default(self):
        """Should create DaskSLURMRunner with defaults."""
        runner = registry.get("dask-slurm")
        assert isinstance(runner, DaskSLURMRunner)
        assert isinstance(runner, BaseRunner)

    @pytest.mark.skipif(not HAS_DASK_JOBQUEUE, reason="dask-jobqueue not installed")
    def test_create_dask_slurm_with_cores(self):
        """Should create DaskSLURMRunner with specified cores."""
        runner = registry.get("dask-slurm", {"cores": 4})
        assert isinstance(runner, DaskSLURMRunner)

    @pytest.mark.skipif(not HAS_DASK_JOBQUEUE, reason="dask-jobqueue not installed")
    def test_create_dask_slurm_with_memory(self):
        """Should create DaskSLURMRunner with specified memory."""
        runner = registry.get("dask-slurm", {"memory": "16GB"})
        assert isinstance(runner, DaskSLURMRunner)

    @pytest.mark.skipif(not HAS_DASK_JOBQUEUE, reason="dask-jobqueue not installed")
    def test_create_dask_slurm_with_queue(self):
        """Should create DaskSLURMRunner with specified queue."""
        runner = registry.get("dask-slurm", {"queue": "gpu"})
        assert isinstance(runner, DaskSLURMRunner)

    @pytest.mark.skipif(not HAS_DASK_JOBQUEUE, reason="dask-jobqueue not installed")
    def test_create_dask_slurm_with_all_params(self):
        """Should create DaskSLURMRunner with all parameters."""
        params = {
            "cores": 4,
            "memory": "16GB",
            "processes": 2,
            "queue": "gpu",
            "walltime": "02:00:00",
            "num_jobs": 10,
            "account": "myaccount",
            "project": "myproject",
        }
        runner = registry.get("dask-slurm", params)
        assert isinstance(runner, DaskSLURMRunner)

    def test_create_runner_unknown_type(self):
        """Should raise ValueError for unknown runner type."""
        with pytest.raises(ValueError, match="Unknown runner type"):
            registry.get("unknown-runner")

    def test_create_runner_unknown_type_suggests_valid(self):
        """Error message should suggest valid runner types."""
        with pytest.raises(ValueError, match="Valid options"):
            registry.get("invalid")

    def test_create_runner_none_params(self):
        """Should handle None params as empty dict."""
        runner = registry.get("sequential", None)
        assert isinstance(runner, SequentialRunner)

    def test_create_runner_empty_params(self):
        """Should handle empty params dict."""
        runner = registry.get("sequential", {})
        assert isinstance(runner, SequentialRunner)


class TestRunnerRegistryRegister:
    """Tests for register method."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Save and restore registry state."""
        original = registry._registry.copy()
        yield
        registry._registry.clear()
        registry._registry.update(original)

    def test_register_non_baserunner_raises_error(self):
        """Test registering non-BaseRunner class raises TypeError."""
        class NotARunner:
            pass

        with pytest.raises(TypeError, match="must inherit from BaseRunner"):
            registry.register("not-runner", NotARunner)

    def test_register_same_name_different_class_raises_error(self):
        """Test registering different class with same name raises ValueError."""
        class AnotherSequential(BaseRunner):
            def resolve(self, data, feature_space):
                pass

        with pytest.raises(ValueError, match="already registered"):
            registry.register("sequential", AnotherSequential)

    def test_register_same_class_twice_succeeds(self):
        """Test re-registering the same class succeeds silently."""
        registry.register("sequential", SequentialRunner)
        assert "sequential" in registry


class TestRunnerRegistryGetName:
    """Tests for get_name method."""

    def test_get_name_registered_class(self):
        """Test get_name returns name for registered class."""
        name = registry.get_name(SequentialRunner)
        assert name == "sequential"

    def test_get_name_unregistered_class(self):
        """Test get_name returns None for unregistered class."""
        class UnregisteredRunner(BaseRunner):
            def resolve(self, data, feature_space):
                pass

        name = registry.get_name(UnregisteredRunner)
        assert name is None


class TestRunnerRegistryIsRegistered:
    """Tests for is_registered method."""

    def test_is_registered_true(self):
        """Test is_registered returns True for registered name."""
        assert registry.is_registered("sequential") is True

    def test_is_registered_false(self):
        """Test is_registered returns False for unregistered name."""
        assert registry.is_registered("nonexistent") is False


class TestRunnerRegistryUnregister:
    """Tests for unregister method."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Save and restore registry state."""
        original = registry._registry.copy()
        yield
        registry._registry.clear()
        registry._registry.update(original)

    def test_unregister_removes_runner(self):
        """Test unregister removes the runner."""
        # Arrange: Verify sequential is registered
        assert "sequential" in registry

        # Act: Unregister
        registry.unregister("sequential")

        # Assert: Removed
        assert "sequential" not in registry

    def test_unregister_nonexistent_raises_keyerror(self):
        """Test unregister raises KeyError for non-existent name."""
        with pytest.raises(KeyError, match="not found in registry"):
            registry.unregister("nonexistent-runner")


class TestRunnerRegistryClear:
    """Tests for clear method."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Save and restore registry state."""
        original = registry._registry.copy()
        yield
        registry._registry.clear()
        registry._registry.update(original)

    def test_clear_removes_all_runners(self):
        """Test clear removes all runners."""
        assert len(registry) > 0
        registry.clear()
        assert len(registry) == 0


class TestRunnerRegistryDunderMethods:
    """Tests for dunder methods."""

    def test_contains_registered(self):
        """Test __contains__ returns True for registered name."""
        assert "sequential" in registry

    def test_contains_unregistered(self):
        """Test __contains__ returns False for unregistered name."""
        assert "nonexistent" not in registry

    def test_iter_yields_names(self):
        """Test __iter__ yields runner names."""
        names = list(registry)
        assert "sequential" in names
        assert isinstance(names[0], str)
