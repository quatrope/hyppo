"""Execution engines for feature extraction workflows."""

from typing import Any

from .base import BaseRunner
from .dask import (
    DaskProcessesRunner,
    DaskRunner,
    DaskSLURMRunner,
    DaskThreadsRunner,
)
from .local_process import LocalProcessRunner
from .registry import registry
from .sequential import SequentialRunner

__all__ = [
    "BaseRunner",
    "DaskProcessesRunner",
    "DaskRunner",
    "DaskSLURMRunner",
    "DaskThreadsRunner",
    "LocalProcessRunner",
    "SequentialRunner",
    "registry",
]


# =============================================================================
# RUNNER REGISTRATION
# =============================================================================

registry.register("sequential", SequentialRunner)
registry.register("local", LocalProcessRunner)
registry.register("dask-threads", DaskThreadsRunner)
registry.register("dask-processes", DaskProcessesRunner)
registry.register("dask-slurm", DaskSLURMRunner)
