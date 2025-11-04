"""Tests for LocalProcessRunner."""

import numpy as np
import pytest

from hyppo.core import FeatureCollection, FeatureSpace, HSI
from hyppo.extractor.base import Extractor
from hyppo.runner import LocalProcessRunner


class SimpleTestExtractor(Extractor):
    """Simple extractor for testing."""

    def _extract(self, data: HSI, **inputs) -> dict:
        """Extract simple test value."""
        return {"simple_value": float(np.mean(data.reflectance))}


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
        input_value = inputs["simple_input"]["simple_value"]
        return {"dependent_value": input_value * 2}


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


class DataAccessExtractor(Extractor):
    """Extractor that verifies shared memory access."""

    def _extract(self, data: HSI, **inputs) -> dict:
        return {
            "shape": data.shape,
            "mean": float(np.mean(data.reflectance)),
            "std": float(np.std(data.reflectance)),
            "n_bands": data.n_bands,
            "wavelength_range": (
                float(data.wavelengths.min()),
                float(data.wavelengths.max()),
            ),
        }


class PixelModifierExtractor(Extractor):
    """Extractor that modifies a specific pixel to verify shared memory."""

    def __init__(
        self, row: int, col: int, marker_value: float, expected_positions: list
    ):
        """Initialize test extractor."""
        super().__init__()
        self.row = row
        self.col = col
        self.marker_value = marker_value
        self.expected_positions = expected_positions

    def _extract(self, data: HSI, **inputs) -> dict:
        # First, modify our designated pixel
        data.reflectance[self.row, self.col, 0] = self.marker_value

        # Then read all positions to see what other workers have written
        # If shared, we see modifications from workers that ran before us
        observed_values = {}
        for i, (r, c) in enumerate(self.expected_positions):
            observed_values[f"pos_{i}"] = float(data.reflectance[r, c, 0])

        return {
            "modified": True,
            "marker": self.marker_value,
            "row": self.row,
            "col": self.col,
            "observed_values": observed_values,
        }


class TimedExtractor(Extractor):
    """Extractor that records execution time to verify parallelism."""

    def __init__(self, sleep_time: float):
        """Initialize test extractor."""
        super().__init__()
        self.sleep_time = sleep_time

    def _extract(self, data: HSI, **inputs) -> dict:
        import time

        start = time.time()
        time.sleep(self.sleep_time)
        end = time.time()
        return {"start": start, "end": end, "duration": end - start}


class TestLocalProcessRunner:
    """Test cases for LocalProcessRunner class."""

    def test_can_instantiate(self):
        """Test that LocalProcessRunner can be instantiated."""
        # Arrange & Act: Create runner
        runner = LocalProcessRunner(num_workers=2)

        # Assert: Instance created
        assert isinstance(runner, LocalProcessRunner)
        assert runner._num_workers == 2
        assert runner._pool is not None

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_init_with_workers(self):
        """Test LocalProcessRunner.__init__() with specified workers."""
        # Act: Create runner with specified workers
        runner = LocalProcessRunner(num_workers=2)

        # Assert: Runner created
        assert isinstance(runner, LocalProcessRunner)
        assert runner._num_workers == 2

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_init_default_workers(self):
        """Test LocalProcessRunner.__init__() with default worker count."""
        # Act: Create runner without specifying workers
        runner = LocalProcessRunner()

        # Assert: Runner created with CPU count workers
        import multiprocessing as mp

        assert isinstance(runner, LocalProcessRunner)
        assert runner._num_workers == mp.cpu_count()

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_init_invalid_workers(self):
        """Test LocalProcessRunner raises error for invalid workers."""
        # Act & Assert: Invalid worker count
        with pytest.raises(ValueError, match="Invalid number of workers"):
            LocalProcessRunner(num_workers=0)

        with pytest.raises(ValueError, match="Invalid number of workers"):
            LocalProcessRunner(num_workers=-1)

    def test_resolve_single_extractor(self, small_hsi):
        """Test resolving single extractor without dependencies."""
        # Arrange: Create runner and feature space
        runner = LocalProcessRunner(num_workers=2)
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
        runner._pool.close()
        runner._pool.join()

    def test_resolve_with_dependencies(self, small_hsi):
        """Test resolving extractors with dependencies."""
        # Arrange: Create runner with dependent extractors
        runner = LocalProcessRunner(num_workers=2)
        simple = SimpleTestExtractor()
        dependent = DependentTestExtractor()
        fs = FeatureSpace.from_list([simple, dependent])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Verify both extractors executed and dependency passed
        assert len(results) == 2
        assert "simple_test" in results
        assert "dependent_test" in results
        assert "dependent_value" in results["dependent_test"]["data"]
        expected_value = results["simple_test"]["data"]["simple_value"] * 2
        assert (
            results["dependent_test"]["data"]["dependent_value"]
            == expected_value
        )

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_shared_memory_data_access(self, small_hsi):
        """Test that worker processes correctly access shared HSI data."""
        # Arrange: Create runner with data access extractor
        runner = LocalProcessRunner(num_workers=2)
        extractor = DataAccessExtractor()
        fs = FeatureSpace.from_list([extractor])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Verify data was correctly accessed
        result_data = results["data_access"]["data"]
        assert result_data["shape"] == small_hsi.shape
        assert result_data["n_bands"] == small_hsi.n_bands
        assert np.isclose(result_data["mean"], np.mean(small_hsi.reflectance))
        assert np.isclose(result_data["std"], np.std(small_hsi.reflectance))

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_resolve_complex_pipeline(self, small_hsi):
        """Test complex pipeline with multiple extractors."""
        # Arrange: Import real extractors and create pipeline
        from hyppo.extractor import MeanExtractor, PCAExtractor, StdExtractor

        runner = LocalProcessRunner(num_workers=2)
        fs = FeatureSpace.from_list(
            [MeanExtractor(), StdExtractor(), PCAExtractor()]
        )

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: All extractors produced results
        assert len(results) == 3
        assert "mean" in results
        assert "std" in results
        assert "p_c_a" in results

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_resolve_empty_feature_space(self, small_hsi):
        """Test resolving with empty feature space."""
        # Arrange: Create empty feature space
        runner = LocalProcessRunner(num_workers=2)
        fs = FeatureSpace({})

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Empty results
        assert len(results) == 0

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_optional_dependency_with_default(self, small_hsi):
        """Test extractor with optional dependency using default."""
        # Arrange: Create runner with only optional dependency extractor
        runner = LocalProcessRunner(num_workers=2)
        optional = OptionalDependencyExtractor()
        fs = FeatureSpace.from_list([optional])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Default was used
        assert len(results) == 1
        assert "optional_dependency" in results
        assert results["optional_dependency"]["data"]["has_input"] is True

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_optional_dependency_provided(self, small_hsi):
        """Test optional dependency when source is provided."""
        # Arrange: Create runner with optional dependency and source
        runner = LocalProcessRunner(num_workers=2)
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
        runner._pool.close()
        runner._pool.join()

    def test_integration_with_feature_space_extract(self, small_hsi):
        """Test using LocalProcessRunner through FeatureSpace.extract()."""
        # Arrange: Create runner and feature space
        from hyppo.extractor import MeanExtractor, StdExtractor

        runner = LocalProcessRunner(num_workers=2)
        fs = FeatureSpace.from_list([MeanExtractor(), StdExtractor()])

        # Act: Call through FeatureSpace.extract()
        results = fs.extract(small_hsi, runner)

        # Assert: Results produced
        assert len(results) == 2
        assert "mean" in results
        assert "std" in results

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_shared_memory_cleanup(self, small_hsi):
        """Test that shared memory is properly cleaned up after extraction."""
        # Arrange: Create runner and feature space
        runner = LocalProcessRunner(num_workers=2)
        extractor = SimpleTestExtractor()
        fs = FeatureSpace.from_list([extractor])

        # Act: Execute extraction
        results = runner.resolve(small_hsi, fs)

        # Assert: Results produced (cleanup happens in finally block)
        assert len(results) == 1

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_multiple_sequential_extractions(self, small_hsi):
        """Test running multiple extractions sequentially with same runner."""
        # Arrange: Create runner and feature space
        runner = LocalProcessRunner(num_workers=2)
        fs = FeatureSpace.from_list([SimpleTestExtractor()])

        # Act: Run multiple extractions
        results1 = runner.resolve(small_hsi, fs)
        results2 = runner.resolve(small_hsi, fs)

        # Assert: Both extractions successful
        assert len(results1) == 1
        assert len(results2) == 1
        assert (
            results1["simple_test"]["data"] == results2["simple_test"]["data"]
        )

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_hsi_with_mask(self, small_hsi):
        """Test extraction with masked HSI data."""
        # Arrange: Create HSI with custom mask
        mask = np.ones((small_hsi.height, small_hsi.width), dtype=bool)
        mask[0, 0] = False  # Mask one pixel
        masked_hsi = HSI(
            reflectance=small_hsi.reflectance,
            wavelengths=small_hsi.wavelengths,
            mask=mask,
            metadata=small_hsi.metadata,
        )

        runner = LocalProcessRunner(num_workers=2)
        fs = FeatureSpace.from_list([DataAccessExtractor()])

        # Act: Execute extraction
        results = runner.resolve(masked_hsi, fs)

        # Assert: Data accessed correctly with mask
        assert len(results) == 1
        assert results["data_access"]["data"]["shape"] == masked_hsi.shape

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_shared_memory_modifications_are_visible(self, small_hsi):
        """Test modifications to shared memory are visible across workers.

        Since extractors run sequentially in topological order, each
        extractor should see modifications made by previous extractors
        if memory is shared.
        """
        # Arrange: Setup positions and marker values
        positions = [(0, 0), (0, 1), (1, 0)]
        marker_values = [999.0, 888.0, 777.0]
        num_extractors = 3

        # Create extractors that will modify different pixels
        # Make them depend on each other to force sequential execution:
        # modifier_0: no deps
        # modifier_1: depends on modifier_0
        # modifier_2: depends on modifier_1
        extractors = []
        for i in range(num_extractors):
            ext = PixelModifierExtractor(
                row=positions[i][0],
                col=positions[i][1],
                marker_value=marker_values[i],
                expected_positions=positions,
            )
            ext._feature_name_override = f"modifier_{i}"
            extractors.append(ext)

        # Create feature space with dependencies
        config = {
            "modifier_0": (extractors[0], {}),
            "modifier_1": (
                extractors[1],
                {"dummy": "modifier_0"},
            ),  # depends on 0
            "modifier_2": (
                extractors[2],
                {"dummy": "modifier_1"},
            ),  # depends on 1
        }
        fs = FeatureSpace(config)

        runner = LocalProcessRunner(num_workers=num_extractors)

        # Act: Execute extractors - each modifies shared memory sequentially
        results = runner.resolve(small_hsi, fs)

        # Assert: All extractors completed
        assert len(results) == num_extractors

        # Critical test: verify each extractor saw previous modifications
        # Extractor 0: original at pos[1] and pos[2], marker at pos[0]
        # Extractor 1: marker at pos[0] and pos[1], original at pos[2]
        # Extractor 2: all markers

        # Check extractor 0 (first to run)
        obs_0 = results["modifier_0"]["data"]["observed_values"]
        assert (
            obs_0["pos_0"] == marker_values[0]
        ), "Extractor 0 should see its own modification"

        # Check extractor 1 (second to run)
        obs_1 = results["modifier_1"]["data"]["observed_values"]
        assert (
            obs_1["pos_0"] == marker_values[0]
        ), "Extractor 1 should see extractor 0's modification"
        assert (
            obs_1["pos_1"] == marker_values[1]
        ), "Extractor 1 should see its own modification"

        # Check extractor 2 (third to run) - this is the key test
        obs_2 = results["modifier_2"]["data"]["observed_values"]
        assert (
            obs_2["pos_0"] == marker_values[0]
        ), "Extractor 2 should see extractor 0's modification"
        assert (
            obs_2["pos_1"] == marker_values[1]
        ), "Extractor 2 should see extractor 1's modification"
        assert (
            obs_2["pos_2"] == marker_values[2]
        ), "Extractor 2 should see its own modification"

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_parallel_execution_of_independent_extractors(self, small_hsi):
        """Test that independent extractors are executed in parallel."""
        import time

        # Create 3 independent extractors that each sleep for 0.5 seconds
        sleep_time = 0.5
        num_extractors = 3
        extractors = []
        for i in range(num_extractors):
            ext = TimedExtractor(sleep_time=sleep_time)
            ext._feature_name_override = f"timed_{i}"
            extractors.append(ext)

        config = {f"timed_{i}": (ext, {}) for i, ext in enumerate(extractors)}
        fs = FeatureSpace(config)

        runner = LocalProcessRunner(num_workers=num_extractors)

        # Act: Execute extractors
        start_all = time.time()
        results = runner.resolve(small_hsi, fs)
        end_all = time.time()
        total_time = end_all - start_all

        # Assert: If parallel, total time should be ~0.5s (one sleep time)
        # If sequential, it would be ~1.5s (three sleep times)
        # Allow overhead for process creation and synchronization
        assert len(results) == num_extractors

        # With parallel execution, time should be close to sleep_time
        # Allow 100% overhead for process management
        # 1.2 seconds if sequential
        min_sequential_time = sleep_time * num_extractors * 0.8

        msg = (
            f"Execution took {total_time:.3f}s, "
            f"expected < {min_sequential_time:.3f}s for parallel"
        )
        assert total_time < min_sequential_time, msg

        # Cleanup
        runner._pool.close()
        runner._pool.join()

    def test_del_method_cleanup(self):
        """Test that __del__ method properly closes pool."""
        runner = LocalProcessRunner(num_workers=2)

        # Delete runner should close and join pool
        del runner
        # Test passes if no exception is raised

    def test_del_method_without_pool(self):
        """Test that __del__ handles missing _pool attribute."""
        runner = LocalProcessRunner(num_workers=2)

        # Save pool reference and remove _pool attribute
        pool = runner._pool
        delattr(runner, "_pool")

        # Should not raise
        del runner

        # Cleanup the orphaned pool
        pool.close()
        pool.join()

    def test_execute_extractor_with_shared_hsi_function(self, small_hsi):
        """Test worker function _execute_extractor_with_shared_hsi."""
        from hyppo.runner.local_process import (
            _execute_extractor_with_shared_hsi,
        )

        # Create shared memory metadata
        runner = LocalProcessRunner(num_workers=1)
        shm_metadata = runner._create_shared_hsi(small_hsi)

        try:
            # Create an extractor
            extractor = SimpleTestExtractor()

            # Call the worker function directly
            result = _execute_extractor_with_shared_hsi(
                extractor, shm_metadata, {}
            )

            # Assert: Result should be valid
            assert "simple_value" in result
            assert isinstance(result["simple_value"], float)

        finally:
            # Cleanup
            runner._cleanup_shared_hsi(shm_metadata)
            runner._pool.close()
            runner._pool.join()

    def test_execute_extractor_with_shared_hsi_with_inputs(self, small_hsi):
        """Test the worker function with input kwargs."""
        from hyppo.runner.local_process import (
            _execute_extractor_with_shared_hsi,
        )

        # Create shared memory metadata
        runner = LocalProcessRunner(num_workers=1)
        shm_metadata = runner._create_shared_hsi(small_hsi)

        try:
            # Create an extractor with dependencies
            extractor = DependentTestExtractor()

            # Create input kwargs
            input_kwargs = {"simple_input": {"simple_value": 42.0}}

            # Call the worker function directly
            result = _execute_extractor_with_shared_hsi(
                extractor, shm_metadata, input_kwargs
            )

            # Assert: Result should use the input
            assert "dependent_value" in result
            assert result["dependent_value"] == 84.0  # 42 * 2

        finally:
            # Cleanup
            runner._cleanup_shared_hsi(shm_metadata)
            runner._pool.close()
            runner._pool.join()

    def test_reconstruct_hsi_from_shared_function(self, small_hsi):
        """Test the _reconstruct_hsi_from_shared function directly."""
        from hyppo.runner.local_process import _reconstruct_hsi_from_shared

        # Create shared memory metadata
        runner = LocalProcessRunner(num_workers=1)
        shm_metadata = runner._create_shared_hsi(small_hsi)

        try:
            # Call the reconstruction function directly
            reconstructed_hsi, shm_refs = _reconstruct_hsi_from_shared(
                shm_metadata
            )

            # Assert: Reconstructed HSI should match original
            assert reconstructed_hsi.shape == small_hsi.shape
            assert reconstructed_hsi.n_bands == small_hsi.n_bands
            assert len(shm_refs) == 3  # reflectance, wavelengths, mask

            # Verify data matches
            assert np.array_equal(
                reconstructed_hsi.reflectance, small_hsi.reflectance
            )
            assert np.array_equal(
                reconstructed_hsi.wavelengths, small_hsi.wavelengths
            )
            assert np.array_equal(reconstructed_hsi.mask, small_hsi.mask)
            assert reconstructed_hsi.metadata == small_hsi.metadata

            # Close shared memory handles
            for shm in shm_refs:
                shm.close()

        finally:
            # Cleanup
            runner._cleanup_shared_hsi(shm_metadata)
            runner._pool.close()
            runner._pool.join()
