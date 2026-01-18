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
