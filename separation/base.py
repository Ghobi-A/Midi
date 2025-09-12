from __future__ import annotations

"""Base classes and factory for source separation backends."""

from abc import ABC, abstractmethod
from typing import Dict, Type


class BaseSeparator(ABC):
    """Abstract base class for audio separation backends.

    Subclasses register themselves by defining a unique ``name`` attribute.
    ``BaseSeparator.create(name)`` can then be used to instantiate the
    appropriate backend implementation.
    """

    #: registry of available separator implementations
    registry: Dict[str, Type["BaseSeparator"]] = {}

    #: unique name for the backend.  Subclasses must override.
    name: str | None = None

    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        super().__init_subclass__(**kwargs)
        # Register subclass if it defines a name
        name = getattr(cls, "name", None)
        if name:
            BaseSeparator.registry[name] = cls

    @classmethod
    def create(cls, name: str) -> "BaseSeparator":
        """Return a separator instance for the given backend name."""
        try:
            separator_cls = cls.registry[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown separator backend: {name}") from exc
        return separator_cls()

    @abstractmethod
    def separate(self, input_path: str, output_dir: str) -> None:
        """Perform source separation on ``input_path``.

        Implementations should write the separated stems to ``output_dir``.
        """
        raise NotImplementedError
