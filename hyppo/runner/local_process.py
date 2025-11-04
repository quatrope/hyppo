"""Process-based runner with shared memory for efficient data sharing."""

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from hyppo.core import Feature, FeatureCollection, HSI
from .base import BaseRunner


class LocalProcessRunner(BaseRunner):
    """
    Process-based runner using shared memory for efficient HSI data sharing.

    This runner distributes feature extraction tasks across multiple
    worker processes while keeping the HSI data in shared memory to
    minimize RAM usage. All processes access the same HSI data without
    copying.
    """

    def __init__(self, num_workers: int | None = None):
        """
        Create a LocalProcessRunner with specified number of worker processes.

        Args:
            num_workers: Number of worker processes
                        (None = use all available cores)

        Raises:
            ValueError: If num_workers is less than 1
        """
        super().__init__()

        if num_workers is None:
            num_workers = mp.cpu_count()

        if num_workers < 1:
            raise ValueError(f"Invalid number of workers: {num_workers}")

        self._num_workers = num_workers
        self._pool = mp.Pool(processes=num_workers)

    def resolve(self, data: HSI, feature_space) -> FeatureCollection:
        """
        Resolve feature extraction using process pool with shared memory.

        Executes extractors in parallel when possible, respecting
        dependency constraints. Independent extractors are executed
        concurrently in the worker pool.

        Args:
            data: HSI object to process
            feature_space: FeatureSpace instance with feature graph

        Returns:
            FeatureCollection with extraction results
        """
        feature_graph = feature_space.feature_graph

        # Create shared memory for HSI data
        shm_metadata = self._create_shared_hsi(data)

        try:
            results = {}
            extracted_results = {}

            # Group extractors by dependency level for parallel execution
            dependency_levels = self._compute_dependency_levels(feature_graph)

            # Execute each level in parallel
            for level in sorted(dependency_levels.keys()):
                extractor_names = dependency_levels[level]

                # Prepare async tasks for all extractors at this level
                async_results = []
                for extractor_name in extractor_names:
                    extractor = feature_graph.extractors[extractor_name]
                    input_mapping = feature_graph.get_input_mapping_for(
                        extractor_name
                    )

                    # Prepare inputs from previous results
                    input_kwargs = {}
                    for input_name, source_name in input_mapping.items():
                        input_kwargs[input_name] = extracted_results[
                            source_name
                        ]

                    # Get defaults for optional inputs
                    defaults = self._get_defaults_for_extractor(extractor)
                    for input_name, default_extractor in defaults.items():
                        if input_name not in input_kwargs:
                            default_result = default_extractor.extract(data)
                            input_kwargs[input_name] = default_result

                    # Submit to pool asynchronously
                    async_result = self._pool.apply_async(
                        _execute_extractor_with_shared_hsi,
                        (extractor, shm_metadata, input_kwargs),
                    )
                    async_results.append(
                        (
                            extractor_name,
                            extractor,
                            input_mapping,
                            async_result,
                        )
                    )

                # Wait for all extractors at this level to complete
                for (
                    extractor_name,
                    extractor,
                    input_mapping,
                    async_result,
                ) in async_results:
                    result = async_result.get()
                    extracted_results[extractor_name] = result

                    results[extractor_name] = Feature(
                        result, extractor, list(input_mapping.keys())
                    )

        finally:
            # Cleanup shared memory
            self._cleanup_shared_hsi(shm_metadata)

        return FeatureCollection.from_features(results)

    def _compute_dependency_levels(
        self, feature_graph
    ) -> dict[int, list[str]]:
        """
        Compute dependency levels for parallel execution.

        Extractors at the same level can be executed in parallel.
        Level 0 = no dependencies, Level 1 = depends on Level 0, etc.

        Args:
            feature_graph: Feature dependency graph

        Returns:
            Dictionary mapping level number to list of extractor names
        """
        levels = {}
        extractor_levels = {}

        def get_level(extractor_name: str) -> int:
            if extractor_name in extractor_levels:
                return extractor_levels[extractor_name]

            input_mapping = feature_graph.get_input_mapping_for(extractor_name)

            if not input_mapping:
                # No dependencies = level 0
                level = 0
            else:
                # Level is max dependency level + 1
                dependency_levels = [
                    get_level(dep) for dep in input_mapping.values()
                ]
                level = max(dependency_levels) + 1

            extractor_levels[extractor_name] = level
            return level

        # Compute level for each extractor
        for extractor_name in feature_graph.extractors.keys():
            level = get_level(extractor_name)
            if level not in levels:
                levels[level] = []
            levels[level].append(extractor_name)

        return levels

    def _create_shared_hsi(self, data: HSI) -> dict:
        """
        Create shared memory blocks for HSI data arrays.

        Args:
            data: HSI object to share

        Returns:
            Dictionary with shared memory metadata and shm references
        """
        # Create shared memory for reflectance array
        reflectance_shm = SharedMemory(
            create=True, size=data.reflectance.nbytes
        )
        reflectance_shared = np.ndarray(
            data.reflectance.shape,
            dtype=data.reflectance.dtype,
            buffer=reflectance_shm.buf,
        )
        reflectance_shared.flags.writeable = True
        np.copyto(reflectance_shared, data.reflectance)

        # Create shared memory for wavelengths
        wavelengths_shm = SharedMemory(
            create=True, size=data.wavelengths.nbytes
        )
        wavelengths_shared = np.ndarray(
            data.wavelengths.shape,
            dtype=data.wavelengths.dtype,
            buffer=wavelengths_shm.buf,
        )
        wavelengths_shared.flags.writeable = True
        np.copyto(wavelengths_shared, data.wavelengths)

        # Create shared memory for mask
        mask_shm = SharedMemory(create=True, size=data.mask.nbytes)
        mask_shared = np.ndarray(
            data.mask.shape, dtype=data.mask.dtype, buffer=mask_shm.buf
        )
        mask_shared.flags.writeable = True
        np.copyto(mask_shared, data.mask)

        return {
            "reflectance": {
                "name": reflectance_shm.name,
                "shape": data.reflectance.shape,
                "dtype": data.reflectance.dtype,
            },
            "wavelengths": {
                "name": wavelengths_shm.name,
                "shape": data.wavelengths.shape,
                "dtype": data.wavelengths.dtype,
            },
            "mask": {
                "name": mask_shm.name,
                "shape": data.mask.shape,
                "dtype": data.mask.dtype,
            },
            "metadata": data.metadata,
            "_shm_refs": [reflectance_shm, wavelengths_shm, mask_shm],
        }

    def _cleanup_shared_hsi(self, shm_metadata: dict):
        """
        Cleanup shared memory blocks.

        Args:
            shm_metadata: Shared memory metadata dictionary
        """
        # Close and unlink shared memory from the stored references
        if "_shm_refs" in shm_metadata:
            for shm in shm_metadata["_shm_refs"]:
                shm.close()
                shm.unlink()

    def __del__(self):
        """Cleanup pool on deletion."""
        if hasattr(self, "_pool"):
            self._pool.close()
            self._pool.join()


def _execute_extractor_with_shared_hsi(
    extractor, shm_metadata: dict, input_kwargs: dict
):
    """
    Worker function to execute extractor with shared HSI data.

    Args:
        extractor: Extractor instance to execute
        shm_metadata: Shared memory metadata for reconstructing HSI
        input_kwargs: Input arguments from dependencies

    Returns:
        Extraction results
    """
    # Reconstruct HSI from shared memory
    hsi, shm_refs = _reconstruct_hsi_from_shared(shm_metadata)

    # Execute extractor
    result = extractor.extract(hsi, **input_kwargs)

    # Close shared memory handles in child process
    for shm in shm_refs:
        shm.close()

    return result


def _reconstruct_hsi_from_shared(
    shm_metadata: dict,
) -> tuple[HSI, list[SharedMemory]]:
    """
    Reconstruct HSI object from shared memory metadata.

    Args:
        shm_metadata: Dictionary with shared memory names, shapes, and dtypes

    Returns:
        HSI object with arrays backed by shared memory
    """
    # Attach to existing shared memory blocks
    reflectance_shm = SharedMemory(name=shm_metadata["reflectance"]["name"])
    reflectance = np.ndarray(
        shm_metadata["reflectance"]["shape"],
        dtype=shm_metadata["reflectance"]["dtype"],
        buffer=reflectance_shm.buf,
    )

    wavelengths_shm = SharedMemory(name=shm_metadata["wavelengths"]["name"])
    wavelengths = np.ndarray(
        shm_metadata["wavelengths"]["shape"],
        dtype=shm_metadata["wavelengths"]["dtype"],
        buffer=wavelengths_shm.buf,
    )

    mask_shm = SharedMemory(name=shm_metadata["mask"]["name"])
    mask = np.ndarray(
        shm_metadata["mask"]["shape"],
        dtype=shm_metadata["mask"]["dtype"],
        buffer=mask_shm.buf,
    )

    # Create HSI with shared memory arrays
    # Note: We keep the shm objects in scope so they don't get
    # garbage collected
    hsi = HSI(
        reflectance=reflectance,
        wavelengths=wavelengths,
        mask=mask,
        metadata=shm_metadata["metadata"],
    )

    # Return shm references to prevent premature garbage collection
    return hsi, [reflectance_shm, wavelengths_shm, mask_shm]
