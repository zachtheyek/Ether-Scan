"""
Test script for system resource monitoring functionality
Simulates a workload to test CPU, GPU, and RAM monitoring
"""

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf

# Setup logging first (same as main.py)
class StreamToLogger:
    """Redirect stream (stdout/stderr) to logging system"""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def setup_test_logging() -> logging.Logger:
    """Configure logging for test script"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # File handler
    log_path = '/Users/zach/Documents/BL-SETI/cloud_deploy/outputs/test_monitoring.log'
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


logger = setup_test_logging()

# Import config and monitoring after logging is set up
from config import Config
from main import log_system_resources


def setup_test_gpu_config():
    """Configure GPU for testing (simplified version)"""
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU configuration complete: {len(gpus)} GPUs detected")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPUs detected, running on CPU")


def simulate_cpu_workload(duration_seconds=10):
    """Simulate CPU-intensive workload"""
    logger.info(f"Starting CPU workload for {duration_seconds} seconds...")

    end_time = time.time() + duration_seconds
    iteration = 0

    while time.time() < end_time:
        # CPU-intensive computation
        _ = np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)
        iteration += 1

        if iteration % 10 == 0:
            logger.info(f"  CPU workload iteration {iteration}")

    logger.info(f"CPU workload complete: {iteration} iterations")


def simulate_gpu_workload(duration_seconds=10):
    """Simulate GPU-intensive workload using TensorFlow"""
    logger.info(f"Starting GPU workload for {duration_seconds} seconds...")

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        logger.warning("No GPUs available, skipping GPU workload")
        return

    end_time = time.time() + duration_seconds
    iteration = 0

    # Create large tensors for GPU computation
    with tf.device('/GPU:0'):
        while time.time() < end_time:
            # GPU-intensive computation
            a = tf.random.normal([2000, 2000])
            b = tf.random.normal([2000, 2000])
            c = tf.matmul(a, b)
            _ = tf.reduce_sum(c)

            iteration += 1

            if iteration % 5 == 0:
                logger.info(f"  GPU workload iteration {iteration}")

    logger.info(f"GPU workload complete: {iteration} iterations")


def simulate_memory_workload(duration_seconds=10):
    """Simulate RAM-intensive workload"""
    logger.info(f"Starting memory workload for {duration_seconds} seconds...")

    # Allocate large arrays to increase RAM usage
    large_arrays = []

    for i in range(5):
        logger.info(f"  Allocating large array {i+1}/5 (~500MB each)")
        arr = np.random.rand(8000, 8000)  # ~500MB per array
        large_arrays.append(arr)
        time.sleep(duration_seconds / 5)

    logger.info(f"Memory workload complete: {len(large_arrays)} arrays allocated")

    # Keep arrays in memory for a bit
    time.sleep(2)

    # Cleanup
    del large_arrays
    logger.info("Memory arrays deallocated")


def run_test():
    """Run the monitoring test"""
    logger.info("="*80)
    logger.info("SYSTEM RESOURCE MONITORING TEST")
    logger.info("="*80)

    # Setup GPU config
    setup_test_gpu_config()

    # Create config
    config = Config()

    # Override output path for test
    config.output_path = '/Users/zach/Documents/BL-SETI/cloud_deploy/outputs'

    logger.info(f"Output path: {config.output_path}")
    logger.info(f"Plot will be saved to: {config.output_path}/plots/resource_utilization.png")

    # Start resource monitoring
    logger.info("\n" + "="*80)
    monitor = log_system_resources(config)
    logger.info("="*80 + "\n")

    # Wait for monitoring to initialize
    time.sleep(2)

    try:
        # Run different workloads
        logger.info("\n" + "-"*80)
        logger.info("PHASE 1: CPU Workload")
        logger.info("-"*80)
        simulate_cpu_workload(duration_seconds=15)

        # Cool down
        logger.info("\nCooling down for 5 seconds...")
        time.sleep(5)

        logger.info("\n" + "-"*80)
        logger.info("PHASE 2: GPU Workload")
        logger.info("-"*80)
        simulate_gpu_workload(duration_seconds=15)

        # Cool down
        logger.info("\nCooling down for 5 seconds...")
        time.sleep(5)

        logger.info("\n" + "-"*80)
        logger.info("PHASE 3: Memory Workload")
        logger.info("-"*80)
        simulate_memory_workload(duration_seconds=15)

        # Final cool down
        logger.info("\nFinal cool down for 5 seconds...")
        time.sleep(5)

        logger.info("\n" + "="*80)
        logger.info("TEST COMPLETE")
        logger.info("="*80)

    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user (Ctrl+C)")
        logger.info("Resource monitoring will save plot on exit...")

    except Exception as e:
        logger.error(f"\n\nTest failed with error: {e}")
        raise

    finally:
        # Monitor will automatically save on exit via atexit handler
        logger.info("\nExiting test (monitor will save plot automatically)...")


if __name__ == '__main__':
    run_test()
