"""
Main entry point for Etherscan Pipeline
"""

import argparse
import logging
import os
import sys
import numpy as np
import tensorflow as tf
from typing import Optional
import json
import gc
import time
from skimage.transform import downscale_local_mean
from multiprocessing import Pool, cpu_count
import threading
import psutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from datetime import datetime
import atexit
import signal

from config import Config
from training import train_full_pipeline, get_latest_tag

# Global variable to store chunk data for multiprocessing workers
# This avoids serialization overhead when passing data to workers
_GLOBAL_CHUNK_DATA = None

def _init_background_worker(chunk_data):
    """Initialize worker process with chunk data"""
    global _GLOBAL_CHUNK_DATA
    _GLOBAL_CHUNK_DATA = chunk_data


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
        # Flush any remaining content in linebuf if needed
        if self.linebuf:
            self.logger.log(self.level, self.linebuf.rstrip())
            self.linebuf = ''

        # Flush all handlers attached to the logger
        for handler in self.logger.handlers:
            handler.flush()


def setup_logging(log_filepath: str) -> logging.Logger:
    """
    Configure logging to write to both log file & console
    Captures all output sources: Python logging, TensorFlow, warnings, print statements, and stderr
    Must be called after importing TensorFlow to override its default logging config
    """
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Ignore DEBUG level logs
    root_logger.handlers.clear()  # Clear existing handlers

    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    # Setup file handler
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Setup stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Redirect TensorFlow logs to Python logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all TF logs
    tf.get_logger().setLevel(logging.INFO)
    tf_logger = tf.get_logger()
    tf_logger.handlers = []  # Remove TF's default handlers
    tf_logger.propagate = True  # Use root logger handlers

    # Capture Python warnings module output
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.WARNING)

    # Redirect stdout and stderr to logging
    # This captures print statements and C library output
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

    return logging.getLogger(__name__)

# Setup logging immediately after imports to ensure logging is configured before any other code runs
logger = setup_logging('/datax/scratch/zachy/outputs/etherscan/train_pipeline.log')


# class ResourceMonitor:
#     """Background thread to monitor system resources"""
#
#     def __init__(self, interval=1.0):
#         self.interval = interval
#         self.running = False
#         self.thread = None
#
#         # Data storage
#         self.timestamps = []
#         self.cpu_percent_per_core = []  # List of lists, one per core
#         self.ram_percent = []
#         self.gpu_usage = []  # List of lists, one per GPU
#         self.gpu_memory = []  # List of lists, one per GPU
#         self.gpu_names = []
#
#         # Detect GPUs
#         self._detect_gpus()
#
#     def _detect_gpus(self):
#         """Detect available GPUs"""
#         try:
#             gpus = tf.config.list_physical_devices('GPU')
#             self.num_gpus = len(gpus)
#
#             # Try to get GPU names using nvidia-smi if available
#             if self.num_gpus > 0:
#                 try:
#                     import subprocess
#                     result = subprocess.run(
#                         ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
#                         capture_output=True, text=True, timeout=5
#                     )
#                     if result.returncode == 0:
#                         self.gpu_names = [name.strip() for name in result.stdout.strip().split('\n')]
#                     else:
#                         self.gpu_names = [f"GPU:{i}" for i in range(self.num_gpus)]
#                 except:
#                     self.gpu_names = [f"GPU:{i}" for i in range(self.num_gpus)]
#             else:
#                 self.gpu_names = []
#
#         except:
#             self.num_gpus = 0
#             self.gpu_names = []
#
#     def _get_gpu_stats(self):
#         """Get GPU usage and memory statistics"""
#         gpu_utils = []
#         gpu_mems = []
#
#         if self.num_gpus > 0:
#             try:
#                 import subprocess
#                 # Get GPU utilization
#                 result = subprocess.run(
#                     ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
#                      '--format=csv,noheader,nounits'],
#                     capture_output=True, text=True, timeout=5
#                 )
#
#                 if result.returncode == 0:
#                     for line in result.stdout.strip().split('\n'):
#                         parts = line.split(',')
#                         util = float(parts[0].strip())
#                         mem_used = float(parts[1].strip())
#                         mem_total = float(parts[2].strip())
#                         mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
#
#                         gpu_utils.append(util)
#                         gpu_mems.append(mem_percent)
#                 else:
#                     gpu_utils = [0.0] * self.num_gpus
#                     gpu_mems = [0.0] * self.num_gpus
#             except:
#                 gpu_utils = [0.0] * self.num_gpus
#                 gpu_mems = [0.0] * self.num_gpus
#
#         return gpu_utils, gpu_mems
#
#     def _monitor_loop(self):
#         """Background monitoring loop"""
#         start_time = time.time()
#
#         while self.running:
#             try:
#                 # Record timestamp
#                 current_time = time.time() - start_time
#                 self.timestamps.append(current_time)
#
#                 # CPU per core
#                 cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
#                 self.cpu_percent_per_core.append(cpu_per_core)
#
#                 # RAM
#                 ram = psutil.virtual_memory().percent
#                 self.ram_percent.append(ram)
#
#                 # GPU
#                 gpu_utils, gpu_mems = self._get_gpu_stats()
#                 self.gpu_usage.append(gpu_utils)
#                 self.gpu_memory.append(gpu_mems)
#
#                 # Sleep until next interval
#                 time.sleep(self.interval)
#
#             except Exception as e:
#                 logger.error(f"Error in resource monitoring: {e}")
#                 time.sleep(self.interval)
#
#     def start(self):
#         """Start monitoring in background thread"""
#         if not self.running:
#             self.running = True
#             self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
#             self.thread.start()
#
#             # Log system information
#             logger.info("="*60)
#             logger.info("RESOURCE MONITORING STARTED")
#             logger.info(f"  CPU Cores: {psutil.cpu_count()}")
#             logger.info(f"  GPUs Detected: {self.num_gpus}")
#             if self.num_gpus > 0:
#                 for i, name in enumerate(self.gpu_names):
#                     logger.info(f"    GPU {i}: {name}")
#             logger.info(f"  Monitoring Interval: {self.interval}s")
#             logger.info(f"  Output: <config.output_path>/plots/resource_utilization.png")
#             logger.info("="*60)
#
#     def stop(self):
#         """Stop monitoring"""
#         if self.running:
#             self.running = False
#             if self.thread:
#                 self.thread.join(timeout=5)
#
#             duration_mins = self.timestamps[-1] / 60 if self.timestamps else 0
#             logger.info("="*60)
#             logger.info("RESOURCE MONITORING STOPPED")
#             logger.info(f"  Duration: {duration_mins:.2f} minutes")
#             logger.info(f"  Samples Collected: {len(self.timestamps)}")
#             logger.info("="*60)
#
#     def save_plot(self, output_path):
#         """Generate and save resource utilization plot"""
#         if len(self.timestamps) == 0:
#             logger.warning("No monitoring data to plot")
#             return
#
#         # Ensure output directory exists
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#         # Create figure with 3 subplots
#         fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
#         fig.suptitle('System Resource Utilization', fontsize=16, fontweight='bold')
#
#         timestamps = np.array(self.timestamps) / 60  # Convert to minutes
#
#         # ===== CPU Plot =====
#         ax_cpu = axes[0]
#         cpu_data = np.array(self.cpu_percent_per_core).T  # Shape: (num_cores, num_samples)
#         num_cores = cpu_data.shape[0]
#
#         colors = plt.cm.tab20(np.linspace(0, 1, num_cores))
#         for core_idx in range(num_cores):
#             ax_cpu.plot(timestamps, cpu_data[core_idx], label=f'Core {core_idx}',
#                        color=colors[core_idx], linewidth=0.8, alpha=0.7)
#
#         ax_cpu.set_ylabel('CPU Usage (%)', fontsize=12, fontweight='bold')
#         ax_cpu.set_ylim(0, 100)
#         ax_cpu.grid(True, alpha=0.3)
#         ax_cpu.legend(ncol=min(8, num_cores), fontsize=6, loc='upper left')
#         ax_cpu.set_title(f'CPU Usage (per core, n={num_cores})', fontsize=12)
#
#         # ===== RAM Plot =====
#         ax_ram = axes[1]
#         ax_ram.plot(timestamps, self.ram_percent, color='blue', linewidth=1.5)
#         ax_ram.fill_between(timestamps, self.ram_percent, alpha=0.3, color='blue')
#         ax_ram.set_ylabel('RAM Pressure (%)', fontsize=12, fontweight='bold')
#         ax_ram.set_ylim(0, 100)
#         ax_ram.grid(True, alpha=0.3)
#         ax_ram.set_title('RAM Pressure', fontsize=12)
#
#         # ===== GPU Plot =====
#         ax_gpu = axes[2]
#         if self.num_gpus > 0:
#             gpu_usage_data = np.array(self.gpu_usage).T  # Shape: (num_gpus, num_samples)
#             gpu_memory_data = np.array(self.gpu_memory).T  # Shape: (num_gpus, num_samples)
#
#             # Create second y-axis for memory
#             ax_gpu_mem = ax_gpu.twinx()
#
#             colors = plt.cm.tab10(np.linspace(0, 1, self.num_gpus))
#
#             for gpu_idx in range(self.num_gpus):
#                 color = colors[gpu_idx]
#                 gpu_name = self.gpu_names[gpu_idx] if gpu_idx < len(self.gpu_names) else f"GPU:{gpu_idx}"
#
#                 # Usage (solid line, y1)
#                 ax_gpu.plot(timestamps, gpu_usage_data[gpu_idx],
#                            label=f'{gpu_name} Usage',
#                            color=color, linewidth=1.5, alpha=0.9)
#
#                 # Memory (dashed line, y2, dimmer)
#                 ax_gpu_mem.plot(timestamps, gpu_memory_data[gpu_idx],
#                                label=f'{gpu_name} Memory',
#                                color=color, linewidth=1.5, alpha=0.6, linestyle='--')
#
#             ax_gpu.set_ylabel('GPU Usage (%)', fontsize=12, fontweight='bold')
#             ax_gpu_mem.set_ylabel('GPU Memory (%)', fontsize=12, fontweight='bold')
#             ax_gpu.set_ylim(0, 100)
#             ax_gpu_mem.set_ylim(0, 100)
#
#             # Combine legends
#             lines1, labels1 = ax_gpu.get_legend_handles_labels()
#             lines2, labels2 = ax_gpu_mem.get_legend_handles_labels()
#             ax_gpu.legend(lines1 + lines2, labels1 + labels2,
#                          ncol=min(4, self.num_gpus * 2), fontsize=8, loc='upper left')
#         else:
#             ax_gpu.text(0.5, 0.5, 'No GPUs detected',
#                        ha='center', va='center', transform=ax_gpu.transAxes, fontsize=14)
#             ax_gpu.set_ylabel('GPU Usage (%)', fontsize=12, fontweight='bold')
#
#         ax_gpu.grid(True, alpha=0.3)
#         ax_gpu.set_title(f'GPU Usage & Memory (n={self.num_gpus})', fontsize=12)
#         ax_gpu.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
#
#         # Adjust layout and save
#         plt.tight_layout()
#         plt.savefig(output_path, dpi=150, bbox_inches='tight')
#         plt.close(fig)
#
#         logger.info(f"Resource utilization plot saved to: {output_path}")
#
#
# # Global resource monitor instance
# _RESOURCE_MONITOR = None
#
#
# def log_system_resources(config: Config):
#     """
#     Start monitoring CPU, GPU, and RAM usage in background thread.
#     Should be called at the start of main.py execution.
#     Automatically saves plot on exit or keyboard interrupt.
#
#     Args:
#         config: Configuration object containing output_path
#
#     Returns:
#         ResourceMonitor: The monitoring instance
#     """
#     global _RESOURCE_MONITOR
#
#     if _RESOURCE_MONITOR is not None:
#         logger.warning("Resource monitoring already started")
#         return _RESOURCE_MONITOR
#
#     # Create monitor instance
#     _RESOURCE_MONITOR = ResourceMonitor(interval=1.0)
#
#     # Define cleanup function
#     def save_on_exit():
#         if _RESOURCE_MONITOR is not None:
#             _RESOURCE_MONITOR.stop()
#             output_path = os.path.join(config.output_path, 'plots', 'resource_utilization.png')
#             _RESOURCE_MONITOR.save_plot(output_path)
#
#     # Register cleanup handlers
#     atexit.register(save_on_exit)
#
#     # Handle SIGINT (Ctrl+C) and SIGTERM
#     def signal_handler(signum, frame):
#         logger.info(f"Received signal {signum}, saving resource plot...")
#         save_on_exit()
#         sys.exit(0)
#
#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)
#
#     # Start monitoring
#     _RESOURCE_MONITOR.start()
#
#     return _RESOURCE_MONITOR


def setup_gpu_config():
    """Configure GPU memory growth, memory limits, multi-GPU strategy with load balancing & async allocator"""

    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Prevent memory fragmentation within each GPU
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'  # Aggressive cleanup of intermediate tensors

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set equal memory limits for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14000)]  # 14GiB limit per GPU
                )

            # Set distributed strategy to prevent uneven GPU memory usage
            try:
                # Primary choice: NCCL for NVIDIA GPUs
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.NcclAllReduce(num_packs=2)
                )
                logger.info("Using NcclAllReduce for optimal NVIDIA GPU performance")

            except Exception as e:
                # Fallback: HierarchicalCopyAllReduce
                logger.warning(f"NCCL failed ({e}), using HierarchicalCopyAllReduce")
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=2)
                )

            logger.info(f"Distributed strategy: {strategy.num_replicas_in_sync} GPUs")
            return strategy

        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
            return None

    else:
        logger.warning("No GPUs detected, running on CPU")
        return None


def _downsample_cadence_worker(args):
    """
    Worker function to downsample a single cadence in parallel
    Uses global chunk data to avoid serialization overhead

    Args:
        args: Tuple of (cadence_idx, downsample_factor, final_width)

    Returns:
        Downsampled cadence of shape (6, 16, final_width) or None if invalid
    """
    global _GLOBAL_CHUNK_DATA
    cadence_idx, downsample_factor, final_width = args

    # Get cadence from global chunk data
    cadence = _GLOBAL_CHUNK_DATA[cadence_idx]

    # Skip invalid cadences
    if np.any(np.isnan(cadence)) or np.any(np.isinf(cadence)) or np.max(cadence) <= 0:
        return None

    # Downsample each observation separately
    downsampled_cadence = np.zeros((6, 16, final_width), dtype=np.float32)

    for obs_idx in range(6):
        downsampled_cadence[obs_idx] = downscale_local_mean(
            cadence[obs_idx], (1, downsample_factor)
        ).astype(np.float32)

    return downsampled_cadence


def load_background_data(config: Config, n_processes: Optional[int] = None) -> np.ndarray:
    """
    Load & downsample background plates for pipeline using parallel processing

    Args:
        config: Configuration object
        n_processes: Number of parallel processes (defaults to cpu_count())
    """
    logger.info(f"Loading background data from {config.data_path}")

    # Use config values for memory management
    num_target_backgrounds = config.data.num_target_backgrounds
    downsample_factor = config.data.downsample_factor
    final_width = config.data.width_bin // downsample_factor

    chunk_size = config.data.background_load_chunk_size
    max_chunks = config.data.max_chunks_per_file

    logger.info(f"Target backgrounds: {num_target_backgrounds}")
    logger.info(f"Processing chunks of: {chunk_size}")
    logger.info(f"Final resolution: {final_width}")

    # Set number of processes for multiprocessing
    if n_processes is None:
        n_processes = cpu_count()
    logger.info(f"Using {n_processes} workers for parallel background loading")

    all_backgrounds = []

    for filename in config.data.train_files:
        filepath = config.get_training_file_path(filename)

        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            continue

        logger.info(f"Processing {filename}...")

        # Get subset parameters from config
        start, end = config.get_file_subset(filename)

        try:
            # Use memory mapping to avoid loading full file
            raw_data = np.load(filepath, mmap_mode='r')

            # Apply subset if specified in config
            if start is not None or end is not None:
                raw_data = raw_data[start:end]

            logger.info(f"  Raw data shape: {raw_data.shape}")

            # Divide background into equal chunks, then cutoff if exceeds max_chunks
            n_chunks = min(max_chunks, (raw_data.shape[0] + chunk_size - 1) // chunk_size)

            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, raw_data.shape[0])

                # Load chunk into memory
                chunk_data = np.array(raw_data[chunk_start:chunk_end])

                # Process cadences in parallel using multiprocessing
                # Create pool with chunk data initialized in workers to avoid serialization
                with Pool(processes=n_processes, initializer=_init_background_worker,
                         initargs=(chunk_data,)) as pool:

                    # Prepare arguments (just indices, not data - data is in global state)
                    n_cadences = min(chunk_data.shape[0], num_target_backgrounds - len(all_backgrounds))
                    args_list = [
                        (i, downsample_factor, final_width)  # Just pass the chunk index, not the full cadence data
                        for i in range(n_cadences)
                    ]

                    # Process cadences using pool
                    # Calculate optimal chunksize for load balancing
                    # Aim for ~4 chunks per worker to balance overhead vs parallelism
                    chunksize = max(1, n_cadences // (n_processes * 4))
                    results = pool.map(_downsample_cadence_worker, args_list, chunksize=chunksize)

                    # Collect valid results (filter out None from invalid cadences)
                    for result in results:
                        if result is not None:
                            all_backgrounds.append(result)
                            if len(all_backgrounds) >= num_target_backgrounds:
                                break

                # Clear chunk from memory
                del chunk_data
                gc.collect()

            logger.info(f"  Processed {len(all_backgrounds)} cadences so far")

            # Clear raw data reference
            del raw_data
            gc.collect()

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue

    if len(all_backgrounds) == 0:
        raise ValueError("No background data loaded successfully")

    # Stack all backgrounds
    background_array = np.array(all_backgrounds, dtype=np.float32)

    # Sanity check: print descriptive stats
    min_val = np.min(background_array)
    max_val = np.max(background_array)
    mean_val = np.mean(background_array)

    logger.info(f"Total background cadences loaded: {background_array.shape[0]}")
    logger.info(f"Background array shape: {background_array.shape}")
    logger.info(f"Background value range: [{min_val:.6f}, {max_val:.6f}]")
    logger.info(f"Background mean: {mean_val:.6f}")
    logger.info(f"Memory usage: {background_array.nbytes / 1e9:.2f} GB")
    logger.info(f"Background data ready at {background_array.shape[3]} resolution")

    return background_array


def train_command(args):
    """Execute training pipeline with distributed strategy & fault tolerance"""
    logger.info("="*60)
    logger.info("Starting Etherscan Training Pipeline")
    logger.info("="*60)

    # Load configuration
    config = Config()

    # Start resource monitoring
    log_system_resources(config)

    # Setup GPU and get strategy
    strategy = setup_gpu_config()

    # Override config values with CLI args
    if args.num_target_backgrounds:
        config.data.num_target_backgrounds = args.num_target_backgrounds
    if args.background_load_chunk_size:
        config.data.background_load_chunk_size = args.background_load_chunk_size
    if args.max_chunks_per_file:
        config.data.max_chunks_per_file = args.max_chunks_per_file
    if args.train_files:
        config.data.train_files = args.train_files
    if args.rounds:
        config.training.num_training_rounds = args.rounds
    if args.epochs:
        config.training.epochs_per_round = args.epochs
    if args.num_samples_beta_vae:
        config.training.num_samples_beta_vae = args.num_samples_beta_vae
    if args.num_samples_rf:
        config.training.num_samples_rf = args.num_samples_rf
    if args.train_val_split:
        config.training.train_val_split = args.train_val_split
    if args.batch_size:
        config.training.per_replica_batch_size = args.batch_size
    if args.global_batch_size:
        config.training.global_batch_size = args.global_batch_size
    if args.val_batch_size:
        config.training.per_replica_val_batch_size = args.val_batch_size
    if args.signal_injection_chunk_size:
        config.training.signal_injection_chunk_size = args.signal_injection_chunk_size
    if args.snr_base:
        config.training.snr_base = args.snr_base
    if args.initial_snr_range:
        config.training.initial_snr_range = args.initial_snr_range
    if args.final_snr_range:
        config.training.final_snr_range = args.final_snr_range
    if args.curriculum_schedule:
        config.training.curriculum_schedule = args.curriculum_schedule
    if args.exponential_decay_rate:
        config.training.exponential_decay_rate = args.exponential_decay_rate
    if args.step_easy_rounds:
        config.training.step_easy_rounds = args.step_easy_rounds
    if args.step_hard_rounds:
        config.training.step_hard_rounds = args.step_hard_rounds
    if args.base_learning_rate:
        config.training.base_learning_rate = args.base_learning_rate
    if args.min_learning_rate:
        config.training.min_learning_rate = args.min_learning_rate
    if args.min_pct_improvement:
        config.training.min_pct_improvement = args.min_pct_improvement
    if args.patience_threshold:
        config.training.patience_threshold = args.patience_threshold
    if args.reduction_factor:
        config.training.reduction_factor = args.reduction_factor
    if args.max_retries:
        config.training.max_retries = args.max_retries
    if args.retry_delay:
        config.training.retry_delay = args.retry_delay
    if args.load_tag:
        tag = args.load_tag
        if tag.startswith('round_'):
            start_round = int(tag.split('_')[1]) + 1  # Start training from the round proceeding model checkpoint
        else:
            start_round = 1
    else:
        tag = None
        start_round = 1
    if args.load_dir:
        dir = args.load_dir
    else:
        dir = None
    if args.save_tag:
        final_tag = args.save_tag
    else:
        final_tag = None

    logger.info(f"Configuration:")
    logger.info(f"  Number of rounds: {config.training.num_training_rounds}")
    logger.info(f"  Epochs per round: {config.training.epochs_per_round}")
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Output path: {config.output_path}")

    # Load and preprocess background data
    try:
        background_data = load_background_data(config)
    except Exception as e:
        logger.error(f"Failed to load background data: {e}")
        sys.exit(1)

    logger.info(f"Background data loaded: {background_data.shape}")

    # Train models with fault tolerance
    logger.info("Starting training pipeline...")

    max_retries = config.training.max_retries
    retry_delay = config.training.retry_delay

    for attempt in range(max_retries):
        try:
            logger.info(f"Training attempt: {attempt+1}/{max_retries}")

            if attempt > 0:
                logger.info(f"Retrying training from round {start_round}")

            # Reinitialize training pipeline on each attempt so no corrupted state is persisted
            pipeline = train_full_pipeline(
                config,
                background_data,
                strategy=strategy,
                tag=tag,
                dir=dir,
                start_round=start_round,
                final_tag=final_tag
            )

            # If we get here, training succeeded
            break

        except KeyboardInterrupt:
            # Don't retry on user interruption
            logger.info("Training interupted by user")
            raise

        except Exception as e:
            logger.error(f"Training attempt {attempt+1} failed with error: {e}")

            if attempt < max_retries - 1:
                # Retry taining
                logger.info(f"Attempting to recover from failure: attempt {attempt+2}/{max_retries}")

                try:
                    # Clean up failed pipeline
                    if 'pipeline' in locals():
                        del pipeline
                    gc.collect()
                    logger.info("Cleaned up failed pipeline")

                    # Find the latest checkpoint & determine where to resume from
                    dir = 'checkpoints'
                    tag = get_latest_tag(os.path.join(config.model_path, dir))
                    if tag.startswith("round_"):
                        start_round = int(tag.split("_")[1]) + 1  # Start training from the round proceeding model checkpoint
                        logger.info(f"Loaded latest checkpoint from round {start_round-1}")
                    else:
                        logger.info("No valid checkpoints loaded")
                        raise ValueError(f"No valid checkpoints loaded")

                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)

                except Exception as recovery_error:
                    # If no checkpoints loaded, restart from last valid start_round
                    logger.error(f"Recovery failed: {recovery_error}")
                    logger.info(f"Restarting training from round {start_round} in {retry_delay} seconds")
                    time.sleep(retry_delay)

            else:
                # Max retries exceeded
                logger.error(f"Training attempts exceeded maximum retries ({max_retries})")
                logger.error(f"Final error: {e}")
                raise Exception(f"Training attempts exceeded maximum retries ({max_retries}). Final error: {e}")

    # Save configuration
    config_path = os.path.join(config.model_path, f'config_{final_tag}.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")

    logger.info("="*60)
    logger.info("Training completed successfully!")
    logger.info("="*60)


def inference_command(args):
    """Execute inference command"""
    logger.info("Starting inference pipeline...")

    # Setup GPU
    setup_gpu_config()

    # Load configuration
    config = Config()

    # Load saved config if provided
    if args.config:
        with open(args.config, 'r') as f:
            saved_config = json.load(f)
            # Update config with saved values
            for section_key, section_value in saved_config.items():
                if hasattr(config, section_key) and isinstance(section_value, dict):
                    for key, value in section_value.items():
                        if hasattr(getattr(config, section_key), key):
                            setattr(getattr(config, section_key), key, value)

    # Prepare observation files
    observation_files = []

    # Check for prepared test cadences
    test_dir = os.path.join(config.data_path, 'testing', 'prepared_cadences')
    if os.path.exists(test_dir):
        # Load prepared cadences
        for cadence_idx in range(args.n_bands):
            cadence_files = []
            for obs_idx in range(6):
                obs_file = os.path.join(test_dir, f'cadence_{cadence_idx:04d}_obs_{obs_idx}.npy')
                if os.path.exists(obs_file):
                    cadence_files.append(obs_file)

            if len(cadence_files) == 6:
                observation_files.append(cadence_files)

    if not observation_files:
        logger.error("No observation files found. Please prepare test data first.")
        sys.exit(1)

    logger.info(f"Found {len(observation_files)} cadences for inference")

    # Run inference
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = args.output or f"/outputs/seti/detections_{timestamp}.csv"

    results = run_inference(
        config,
        observation_files,
        args.vae_model,
        args.rf_model,
        output_path
    )

    logger.info(f"Inference completed. Results saved to {output_path}")

    # Print summary
    if results is not None and not results.empty:
        n_total = len(results)
        n_high_conf = len(results[results['confidence'] > 0.9])
        logger.info(f"Total detections: {n_total}")
        logger.info(f"High confidence (>90%): {n_high_conf}")


# NOTE: come back to this later
# def evaluate_command(args):
#     """Execute evaluation command"""
#     logger.info("Starting model evaluation...")
#
#     # Setup GPU
#     setup_gpu_config()
#
#     # Load configuration
#     config = Config()
#
#     # Import necessary modules
#     import tensorflow as tf
#     from models.random_forest import RandomForestModel
#
#     # Load models
#     logger.info(f"Loading VAE encoder from {args.vae_model}")
#     vae_encoder = tf.keras.models.load_model(args.vae_model)
#
#     logger.info(f"Loading Random Forest from {args.rf_model}")
#     rf_model = RandomForestModel(config)
#     rf_model.load(args.rf_model)
#
#     # Load or generate test data
#     if args.test_data:
#         logger.info(f"Loading test data from {args.test_data}")
#         test_data = np.load(args.test_data, allow_pickle=True).item()
#     else:
#         logger.info("Generating synthetic test data...")
#         # Load some background for generation
#         background_data = load_background_data(config)
#         generator = DataGenerator(config, background_data[:100])  # Use subset
#         test_data = generator.generate_test_set()
#
#     # Evaluate
#     preprocessor = DataPreprocessor(config)
#
#     # Prepare test data
#     test_true = preprocessor.prepare_batch(test_data['true'])
#     test_false = preprocessor.prepare_batch(test_data['false'])
#
#     # Get predictions
#     _, _, true_latents = vae_encoder.predict(test_true, batch_size=64)
#     _, _, false_latents = vae_encoder.predict(test_false, batch_size=64)
#
#     true_preds = rf_model.predict(true_latents)
#     false_preds = rf_model.predict(false_latents)
#
#     # Calculate metrics
#     tpr = np.mean(true_preds == 1)
#     fpr = np.mean(false_preds == 1)
#     accuracy = np.mean(np.concatenate([true_preds == 1, false_preds == 0]))
#
#     logger.info("="*60)
#     logger.info("Evaluation Results:")
#     logger.info(f"  True Positive Rate: {tpr:.3f}")
#     logger.info(f"  False Positive Rate: {fpr:.3f}")
#     logger.info(f"  Overall Accuracy: {accuracy:.3f}")
#     logger.info("="*60)


# TODO: add assertions to make sure no problematic values gets passed through CLI args
def main():
    """Main entry point to Etherscan pipeline"""
    parser = argparse.ArgumentParser(
        description='Etherscan Pipeline - Search for ETI signals using deep learning'
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Training command
    train_parser = subparsers.add_parser('train', help='Training pipeline (defaults in config.py)')
    train_parser.add_argument('--num-target-backgrounds', type=int, default=None,
                              help='Number of background cadences to load')
    train_parser.add_argument('--background-load-chunk-size', type=int, default=None,
                              help='Maximum cadences to process at once during background loading')
    train_parser.add_argument('--max-chunks-per-file', type=int, default=None,
                              help='Maximum chunks to load from a single file')
    train_parser.add_argument('--train-files', type=str, nargs='+', default=None,
                              help='List of training data files to use')
    train_parser.add_argument('--rounds', type=int, default=None,
                              help='Number of training rounds')
    train_parser.add_argument('--epochs', type=int, default=None,
                              help='Epochs per training round')
    train_parser.add_argument('--num-samples-beta-vae', type=int, default=None,
                              help='Number of training samples for beta-vae')
    train_parser.add_argument('--num-samples-rf', type=int, default=None,
                              help='Number of training samples for random forest')
    train_parser.add_argument('--train-val-split', type=float, default=None,
                              help='Training/validation split for beta-vae')
    train_parser.add_argument('--batch-size', type=int, default=None,
                              help='Per replica batch size for training')
    train_parser.add_argument('--global-batch-size', type=int, default=None,
                              help='Effective batch size for gradient accumulation')
    train_parser.add_argument('--val-batch-size', type=int, default=None,
                              help='Per replica batch size for validation')
    train_parser.add_argument('--signal-injection-chunk-size', type=int, default=None,
                              help='Maximum cadences to process at once during data generation')
    train_parser.add_argument('--snr-base', type=int, default=None,
                              help='Base SNR for curriculum learning')
    train_parser.add_argument('--initial-snr-range', type=int, default=None,
                              help='Initial SNR range for curriculum learning')
    train_parser.add_argument('--final-snr-range', type=int, default=None,
                              help='Final SNR range for curriculum learning')
    train_parser.add_argument('--curriculum-schedule', type=str, default=None,
                              help='Curriculum learning schedule: linear, exponential, or step')
    train_parser.add_argument('--exponential-decay-rate', type=int, default=None,
                              help='Exponential decay rate for curriculum (must be <0)')
    train_parser.add_argument('--step-easy-rounds', type=int, default=None,
                              help='Number of rounds with easy signals (for step schedule)')
    train_parser.add_argument('--step-hard-rounds', type=int, default=None,
                              help='Number of rounds with challenging signals (for step schedule)')
    train_parser.add_argument('--base-learning-rate', type=float, default=None,
                              help='Base learning rate for training')
    train_parser.add_argument('--min-learning-rate', type=float, default=None,
                              help='Minimum learning rate for adaptive LR')
    train_parser.add_argument('--min-pct-improvement', type=float, default=None,
                              help='Minimum percentage improvement for adaptive LR')
    train_parser.add_argument('--patience-threshold', type=int, default=None,
                              help='Consecutive epochs with no improvement before LR reduction')
    train_parser.add_argument('--reduction-factor', type=float, default=None,
                              help='Factor to reduce learning rate by (e.g., 0.2 = 20% reduction)')
    train_parser.add_argument('--max-retries', type=int, default=None,
                              help='Maximum number of retries on training failure')
    train_parser.add_argument('--retry-delay', type=int, default=None,
                              help='Delay in seconds between retries')
    train_parser.add_argument('--load-tag', type=str, default=None,
                              help='Model tag to resume training from. Accepted formats: final_vX, round_XX, YYYYMMDD_HHMMSS')
    train_parser.add_argument('--load-dir', type=str, default=None,
                              help='Directory to load model tag from. Argument appended to outputs directory')
    train_parser.add_argument('--save-tag', type=str, default=None,
                              help='Model tag to save final model. Accepted formats: final_vX, round_XX, YYYYMMDD_HHMMSS')
    # TODO: finish adding train_command args
    # train_parser.add_argument('--start-round', type=int, default=None,
    #                           help='Training round to start from (default: 1, or the next round proceeding checkpoint tag if provided)')

    # NOTE: come back to this later
    # # Inference command
    # inf_parser = subparsers.add_parser('inference', help='Run inference on data')
    # inf_parser.add_argument('vae_model', type=str, help='Path to VAE encoder model')
    # inf_parser.add_argument('rf_model', type=str, help='Path to Random Forest model')
    # inf_parser.add_argument('--config', type=str, help='Path to saved config file')
    # inf_parser.add_argument('--n-bands', type=int, default=16,
    #                       help='Number of frequency bands to process')
    # inf_parser.add_argument('--output', type=str, help='Output file path')

    # NOTE: come back to this later
    # # Evaluation command
    # eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    # eval_parser.add_argument('vae_model', type=str, help='Path to VAE encoder model')
    # eval_parser.add_argument('rf_model', type=str, help='Path to Random Forest model')
    # eval_parser.add_argument('--test-data', type=str, help='Path to test data')

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'inference':
        inference_command(args)
    # elif args.command == 'evaluate':
    #     evaluate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
