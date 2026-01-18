"""Runner registry system for execution engines."""

from typing import Dict, List, Optional, Type

from .base import BaseRunner


class RunnerRegistry:
    """
    Singleton registry for execution runners.

    Maps runner type names to their corresponding classes and provides
    functionality to register new runners dynamically.
    """

    _instance: Optional["RunnerRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "RunnerRegistry":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the runner registry."""
        if not self._initialized:
            self._registry: Dict[str, Type[BaseRunner]] = {}
            self._initialized = True

    def register(self, name: str, runner_class: Type[BaseRunner]) -> None:
        """
        Register a runner class with a specific name.

        Args:
            name: The runner type name (e.g., 'sequential', 'dask-threads')
            runner_class: The runner class to register

        Raises:
            TypeError: If runner_class doesn't inherit from BaseRunner
            ValueError: If name is already registered with a different class
        """
        if not issubclass(runner_class, BaseRunner):
            raise TypeError(
                f"Class {runner_class.__name__} must inherit from BaseRunner"
            )

        if name in self._registry:
            existing_class = self._registry[name]
            if existing_class != runner_class:
                raise ValueError(
                    f"Runner '{name}' already registered "
                    f"with class {existing_class.__name__}. "
                    f"Cannot register {runner_class.__name__}"
                )
            return

        self._registry[name] = runner_class

    def get(self, name: str, params: Optional[Dict] = None) -> BaseRunner:
        """
        Get a runner instance by name.

        Args:
            name: The runner type name
            params: Optional parameters to pass to runner constructor

        Returns:
            An instantiated runner

        Raises:
            ValueError: If the runner is not registered
        """
        if name not in self._registry:
            valid_runners = ", ".join(self.list_runners())
            raise ValueError(
                f"Unknown runner type: '{name}'. "
                f"Valid options: {valid_runners}"
            )

        runner_class = self._registry[name]

        # Handle None params as empty dict
        if params is None:
            params = {}

        return runner_class(**params)

    def get_name(self, runner_class: Type[BaseRunner]) -> Optional[str]:
        """
        Get the registered name for a runner class.

        Args:
            runner_class: The runner class

        Returns:
            The registered name or None if not found
        """
        for name, cls in self._registry.items():
            if cls == runner_class:
                return name
        return None

    def is_registered(self, name: str) -> bool:
        """Check if a runner name is registered."""
        return name in self._registry

    def list_runners(self) -> List[str]:
        """Get a list of all registered runner names."""
        return list(self._registry.keys())

    def unregister(self, name: str) -> None:
        """
        Unregister a runner.

        Args:
            name: The runner type name to unregister

        Raises:
            KeyError: If the runner is not registered
        """
        if name not in self._registry:
            raise KeyError(f"Runner '{name}' not found in registry")
        del self._registry[name]

    def clear(self) -> None:
        """Clear all registered runners."""
        self._registry.clear()

    def __len__(self) -> int:
        """Get the number of registered runners."""
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """Check if a runner is registered (supports 'in' operator)."""
        return self.is_registered(name)

    def __iter__(self):
        """Iterate over runner names."""
        return iter(self._registry.keys())


# Global registry instance
registry = RunnerRegistry()
