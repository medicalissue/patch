"""
TRAP package initialization.

The package exposes high-level entry points for training, detection,
and evaluation modules used throughout the TRAP pipeline.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "data",
    "detection",
    "evaluation",
    "features",
    "models",
    "pipeline",
    "training",
    "visualization",
]

try:
    __version__ = version("trap")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
