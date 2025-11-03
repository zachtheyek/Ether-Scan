"""
Logger package for Aetherscan pipeline
"""

from .logger import (
    init_logger,
    shutdown_logger,
)

__all__ = [
    "init_logger",
    "shutdown_logger",
]
