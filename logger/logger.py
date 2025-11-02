"""
Logger for Aetherscan Pipeline
Uses queue-based logging to avoid deadlocks and corrupted output from worker processes
"""

import logging
import os
import sys
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

import tensorflow as tf

# NOTE: is this needed?
logger = logging.getLogger(__name__)


class StreamToLogger:
    """Redirect stream (stdout/stderr) to logging system"""

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


def setup_logging(log_filepath: str) -> tuple[logging.Logger, Queue, QueueListener]:
    """
    Configure logging to write to both log file & console with multiprocessing support
    Captures all output sources: Python logging, TensorFlow, warnings, print statements, and stderr
    Must be called after importing TensorFlow to override its default logging config

    Workers send log messages to a queue, which a listener thread in the main process
    reads from and writes to the actual log handlers.

    Returns:
        Tuple of (logger, log_queue, listener)
        - logger: Main logger instance
        - log_queue: Queue for worker processes to send log messages
        - listener: QueueListener that must be stopped on exit
    """
    # Create queue for worker processes (no size limit)
    log_queue = Queue(-1)

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
    listener = QueueListener(log_queue, file_handler, stream_handler, respect_handler_level=True)
    listener.start()

    # Add queue handler to root logger (both main and workers use this)
    queue_handler = QueueHandler(log_queue)
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
    # Note that workers will reset these to avoid inheritance issues
    sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

    return logging.getLogger(__name__), log_queue, listener
