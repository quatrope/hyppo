from typing import Any, Dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Enable dot notation access for getting values."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """Enable dot notation access for setting values."""
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """Enable dot notation access for deleting values."""
        try:
            del self[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a regular dictionary."""
        return dict(self)
