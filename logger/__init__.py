"""
Logger package for Aetherscan pipeline
"""

from .logger import (
    get_log_queue,
    get_logger,
    init_logger,
    init_worker_logging,
    shutdown_logger,
)

__all__ = [
    "get_log_queue",
    "get_logger",
    "init_logger",
    "init_worker_logging",
    "shutdown_logger",
]
