from ._dask import DaskRunner
from dask.distributed import LocalCluster, Client


class ProcessRunner(DaskRunner):
    """
    ProcessRunner using Dask distributed for complete parallel execution.
    
    This runner builds a complete Dask computation graph representing all
    feature extractions and their dependencies, then executes the entire
    graph at once using distributed processes, allowing for maximum
    parallelization and memory isolation.
    """

    def __init__(
        self, 
        num_workers: int | None = None, 
        threads_per_worker: int = 1,
        memory_limit: str = "auto"
    ) -> None:
        super().__init__()
        
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"Invalid number of workers: {num_workers}")
        if threads_per_worker < 1:
            raise ValueError(f"Invalid threads per worker: {threads_per_worker}")
        
        self.num_workers = num_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit

    def _setup_cluster(self):
        """Setup Dask distributed cluster optimized for processes."""
        if self._cluster is None:
            self._cluster = LocalCluster(
                n_workers=self.num_workers,
                threads_per_worker=self.threads_per_worker,
                processes=True,
                memory_limit=self.memory_limit,
                silence_logs=True
            )
            self._client = Client(self._cluster)