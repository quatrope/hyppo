"""Extractor registry system for feature extractors."""

from typing import Dict, List, Optional, Type

from .base import Extractor


class ExtractorRegistry:
    """
    Singleton registry for feature extractors.

    Maps extractor names to their corresponding classes and provides
    functionality to register new extractors dynamically.
    """

    _instance: Optional["ExtractorRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "ExtractorRegistry":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the extractor registry."""
        if not self._initialized:
            self._registry: Dict[str, Type[Extractor]] = {}
            self._initialized = True

    def register(self, extractor_class: Type[Extractor]) -> None:
        """
        Register an extractor class.

        Parameters
        ----------
        extractor_class : Type[Extractor]
            The extractor class to register.
        """
        if not issubclass(extractor_class, Extractor):
            raise TypeError(
                f"Class {extractor_class.__name__} must inherit from Extractor"
            )

        extractor_name = extractor_class.__name__

        if extractor_name in self._registry:
            existing_class = self._registry[extractor_name]
            if existing_class != extractor_class:
                raise ValueError(
                    f"Extractor '{extractor_name}' already registered "
                    f"with class {existing_class.__name__}. "
                    f"Cannot register {extractor_class.__name__}"
                )
            # If same class, just return without re-registering
            return

        self._registry[extractor_name] = extractor_class

    def get(self, name: str) -> Type[Extractor]:
        """
        Get an extractor class by name.

        Parameters
        ----------
        name : str
            The name of the extractor.

        Returns
        -------
        Type[Extractor]
            The extractor class.

        Raises
        ------
        KeyError
            If the extractor is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"Extractor '{name}' not found in registry")
        return self._registry[name]

    def is_registered(self, extractor: str | Type[Extractor]) -> bool:
        """Check if an extractor is registered."""
        if isinstance(extractor, str):
            return extractor in self._registry

        return extractor.__name__ in self._registry

    def list_extractors(self) -> List[str]:
        """Get a list of all registered extractor names."""
        return list(self._registry.keys())

    def unregister(self, name: str) -> None:
        """
        Unregister an extractor.

        Parameters
        ----------
        name : str
            The name of the extractor to unregister.

        Raises
        ------
        KeyError
            If the extractor is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"Extractor '{name}' not found in registry")
        del self._registry[name]

    def clear(self) -> None:
        """Clear all registered extractors."""
        self._registry.clear()

    def __len__(self) -> int:
        """Get the number of registered extractors."""
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """Check if an extractor is registered (supports 'in' operator)."""
        return self.is_registered(name)

    def __iter__(self):
        """Iterate over extractor names."""
        return iter(self._registry.keys())


# Global registry instance
registry = ExtractorRegistry()
