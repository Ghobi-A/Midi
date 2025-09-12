"""Source separation utilities."""

from .base import BaseSeparator


def get_separator(name: str) -> BaseSeparator:
    """Return a separator instance for the given backend name."""
    return BaseSeparator.create(name)


# Import known backends so that they register with the factory.  Additional
# backends can be added by simply importing them elsewhere in the project.
from . import demucs_wrapper, spleeter_wrapper  # noqa: F401

__all__ = ["BaseSeparator", "get_separator"]
