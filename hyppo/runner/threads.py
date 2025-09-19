from ._dask import DaskRunner
from dask.distributed import LocalCluster, Client


class ThreadsRunner(DaskRunner):
    """
    ThreadsRunner using Dask distributed with LocalCluster for thread-based execution.
    
    This runner builds a complete Dask computation graph representing all
    feature extractions and their dependencies, then executes the entire
    graph using a local distributed cluster with thread-based workers.
    """

    def __init__(self, num_threads: int | None = None) -> None:
        super().__init__()
        if num_threads is not None and num_threads < 1:
            raise ValueError(f"Invalid number of threads: {num_threads}")
        self.num_threads = num_threads

    def _setup_cluster(self):
        """Setup Dask distributed cluster optimized for threads."""
        if self._cluster is None:
            self._cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=self.num_threads,
                processes=False,    
                memory_limit='auto',
                silence_logs=True
            )
            self._client = Client(self._cluster)