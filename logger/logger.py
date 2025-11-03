"""
Logger for Aetherscan Pipeline
Uses thread-safe queue-based logging to avoid deadlocks and corrupted outputs from
multiple worker processes
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

import tensorflow as tf

logger = logging.getLogger(__name__)

# Global singleton logger instance
_LOGGER = None


class StreamToLogger:
    """Redirect stream (stdout/stderr) to main logging system"""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        # Flush any remaining content in linebuf if needed
        if self.linebuf:
            self.logger.log(self.level, self.linebuf.rstrip())
            self.linebuf = ""

        # Flush all handlers attached to the logger
        for handler in self.logger.handlers:
            handler.flush()


class Logger:
    """
    Thread-safe logging system with multiprocessing support

    Architecture:
    - Main process runs a QueueListener in a background thread
    - Worker processes send log messages to a shared queue
    - Listener consumes from queue and writes to file/console
    - Eliminates concurrent write issues and corrupted outputs
    """

    def __init__(self, log_filepath: str):
        """
        Initialize logger

        Args:
            log_filepath: Path to log file
        """
        self.log_filepath = log_filepath

        # Create queue for worker processes (no size limit)
        self.log_queue = Queue(-1)

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Ignore DEBUG level logs
        root_logger.handlers.clear()  # Clear existing handlers

        # Create formatter
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

        # Setup file handler (only used by main process via listener)
        file_handler = logging.FileHandler(log_filepath, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Setup stream handler (only used by main process via listener)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # Create queue listener - runs in background thread, writes logs from queue
        self.log_listener = QueueListener(
            self.log_queue, file_handler, stream_handler, respect_handler_level=True
        )
        self.log_listener.start()

        # Add queue handler to root logger (both main and workers use this)
        queue_handler = QueueHandler(self.log_queue)
        root_logger.addHandler(queue_handler)

        # Redirect TensorFlow logs to Python logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Show all TF logs
        tf.get_logger().setLevel(logging.INFO)
        tf_logger = tf.get_logger()
        tf_logger.handlers = []  # Remove TF's default handlers
        tf_logger.propagate = True  # Use root logger handlers

        # Capture Python warnings module output
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.WARNING)

        # Redirect stdout and stderr to logging
        # This captures print statements and C library output
        # Note that workers will reset these with init_worker_logging to avoid inheritance issues
        sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

        logger.info(f"Logger initialized at: {log_filepath}")

    def stop(self):
        """Stop the queue listener thread"""
        if self.log_listener is not None:
            self.log_listener.stop()
            logger.info("Logger stopped")


def init_logger(log_filepath: str) -> Logger:
    """
    Initialize global logger instance (call once at startup)

    Args:
        log_filepath: Path to log file

    Returns:
        Logger instance
    """
    global _LOGGER

    if _LOGGER is not None:
        logger.warning("Logger instance already initialized")
        return _LOGGER

    _LOGGER = Logger(log_filepath)

    return _LOGGER


def init_worker_logging():
    """
    Initialize logging for multiprocessing workers.

    Resets stdout/stderr to avoid inherited StreamToLogger from parent
    and configures queue-based logging for process-safe logging.

    Args:
        log_queue: Queue for sending log messages to main process (optional)
    """
    if _LOGGER is None:
        logger.warning(
            "No logger instance initialized - disabling worker logging to avoid conflicts"
        )
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(logging.NullHandler())
        return

    log_queue = _LOGGER.log_queue

    # Reset stdout/stderr to avoid inherited StreamToLogger from parent
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    # Configure process-local logging to use queue
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(QueueHandler(log_queue))
    root_logger.setLevel(logging.INFO)


def get_logger() -> Logger | None:
    """Get the global logger instance"""
    if _LOGGER is None:
        logger.warning("No logger instance initialized")

    return _LOGGER


def shutdown_logger():
    """Shutdown the global logger instance (call on exit)"""
    global _LOGGER

    if _LOGGER is None:
        logger.warning("No logger instance initialized")
        return

    _LOGGER.stop()
    _LOGGER = None
