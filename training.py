"""
Training pipeline for SETI ML models
"""

import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple
import logging
import os
import shutil
import glob
import re
from datetime import datetime
import matplotlib.pyplot as plt
import gc
import psutil
import subprocess

from config import TrainingConfig
from preprocessing import DataPreprocessor
from data_generation import DataGenerator
from models.vae import create_vae_model
from models.random_forest import RandomForestModel

logger = logging.getLogger(__name__)

# NOTE: move assertions to main.py
# class MaxRoundsExceededError(Exception):
#     """
#     Raised when current_round > self.config.training.num_training_rounds during iterative training.
#     Immediately terminates iterative training
#     """
#     pass

def log_system_resources():
    """Log system resource usage"""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / 1e9
    memory_total_gb = memory.total / 1e9
    
    # GPU usage (if available)
    gpu_info = []
    try:
        # Try nvidia-smi for GPU info
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) == 3:
                    gpu_util, mem_used, mem_total = parts
                    gpu_info.append(f"GPU{i}: {gpu_util}% util ({mem_used}MB/{mem_total}MB)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        gpu_info = ["GPU info unavailable"]
    
    resource_str = (f"Resources -- CPU: {cpu_percent:.1f}%, "
                   f"RAM: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({memory.percent:.1f}%), "
                   f"{', '.join(gpu_info)}")
    
    return resource_str

def archive_directory(base_dir: str, target_dirs: Optional[List[str]] = None, round_num: int = 1):
    """
    Archive and clean up a directory
    
    Args:
        base_dir: Base directory to archive/clean
        target_dirs: List of subdirectory names to include in archiving (e.g., ['train', 'validation'])
                    If None, only files are considered (directories are ignored)
        round_num: Training round number (1 for fresh run, >1 for resume)
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Check if base_dir is empty
    is_empty = True
    
    if target_dirs is None:
        # Check for files only (ignore all directories)
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isfile(item_path):
                is_empty = False
                break
    else:
        # Check for files AND target directories
        has_files = False
        has_target_dirs = False
        
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isfile(item_path):
                has_files = True
            elif os.path.isdir(item_path) and item in target_dirs:
                has_target_dirs = True
        
        is_empty = not (has_files or has_target_dirs)
    
    # If empty, do nothing & return
    if is_empty:
        logger.info(f"Directory {base_dir} is empty, nothing to archive")
        return
    
    # Otherwise, archive and clean up
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = os.path.join(base_dir, 'archive', timestamp)
    os.makedirs(archive_dir, exist_ok=True)
    
    if round_num == 1:
        # Fresh run: move everything to archive
        logger.info(f"Archiving the following items from {base_dir}:")
        
        items_moved = 0
        for item in os.listdir(base_dir):
            if item == 'archive':  # Don't move the archive directory itself
                continue
            
            item_path = os.path.join(base_dir, item)
            
            # Move all files
            if os.path.isfile(item_path):
                shutil.move(item_path, os.path.join(archive_dir, item))
                logger.info(f"  {item_path}")
                items_moved += 1
            # Move target directories if specified
            elif os.path.isdir(item_path) and (target_dirs is not None and item in target_dirs):
                shutil.move(item_path, os.path.join(archive_dir, item))
                logger.info(f"  {item_path}")
                items_moved += 1

                # Replace directory with empty one after moving
                os.makedirs(item_path)  
        
        logger.info(f"Moved {items_moved} items to archive: {archive_dir}")
    
    else:
        # Resume: copy to archive, then delete files with round >= round_num
        logger.info(f"Backing up the following items from {base_dir}:")
        
        items_copied = 0
        for item in os.listdir(base_dir):
            if item == 'archive':  # Don't copy the archive directory itself
                continue
            
            item_path = os.path.join(base_dir, item)
            
            # Copy all files
            if os.path.isfile(item_path):
                shutil.copy2(item_path, os.path.join(archive_dir, item))
                logger.info(f"  {item_path}")
                items_copied += 1
            # Copy target directories if specified
            elif os.path.isdir(item_path) and (target_dirs is not None and item in target_dirs):
                shutil.copytree(item_path, os.path.join(archive_dir, item))
                logger.info(f"  {item_path}")
                items_copied += 1

                # TODO: instead of deleting the whole event files, intelligently parse & filter out future steps, then write filtered events to new files
                # Replace directory with empty one after copying
                shutil.rmtree(item_path)
                os.makedirs(item_path, exist_ok=True)
        
        logger.info(f"Backed up {items_copied} items to archive: {archive_dir}")
        
        # Delete files matching "round_X" where X >= round_num
        logger.info(f"Deleting the following items from {base_dir}:")
        pattern = re.compile(r'round_(\d+)')
        deleted_files = []
        
        for item in os.listdir(base_dir):
            if item == 'archive':  # Don't touch the archive directory
                continue
            
            item_path = os.path.join(base_dir, item)
            
            # Only process files, not directories
            if os.path.isfile(item_path):
                match = pattern.search(item)
                if match:
                    round_x = int(match.group(1))
                    if round_x >= round_num:
                        os.remove(item_path)
                        deleted_files.append(item)
                        logger.info(f"  {item_path}")
        
        if deleted_files:
            logger.info(f"Deleted {len(deleted_files)} files with round >= {round_num}")
        else:
            logger.info(f"No files with round >= {round_num} found to delete")

def calculate_curriculum_snr(round_idx: int, total_rounds: int, config: TrainingConfig) -> Tuple[int, int]:
    """
    Calculate SNR parameters for curriculum learning
    
    Args:
        round_idx: Current training round (0-indexed)
        total_rounds: Total number of training rounds
        config: Training configuration
        
    Returns:
        (snr_base, snr_range) tuple
    """
    # Progress through curriculum: 0.0 (easy) -> 1.0 (hard)
    progress = round_idx / (total_rounds - 1)
    
    if config.curriculum_schedule == "linear":
        # Linear progression from wide to narrow SNR range
        current_range = config.initial_snr_range - progress * (config.initial_snr_range - config.final_snr_range)
    elif config.curriculum_schedule == "exponential":
        # Exponential decay - start easy, then get hard quickly
        current_range = config.final_snr_range + (config.initial_snr_range - config.final_snr_range) * np.exp(config.exponential_decay_rate * progress)
    elif config.curriculum_schedule == "step":
        # Step function - easy for first part, hard for second part
        # TODO: add mechanism for more step changes
        if round_idx < config.easy_rounds:
            current_range = config.initial_snr_range
        else:
            current_range = config.final_snr_range
    else:
        raise ValueError(f"'{config.curriculum_schedule} is invalid. Accepted values: 'linear', 'exponential', 'step'")
    
    return config.snr_base, int(current_range)

# NOTE: artifact of moving fault tolerance logic from iterative_training() to train_command()
# TODO: clean up "finding latest tag" instances with this function (e.g. in load_models())
def get_latest_tag(checkpoints_dir: str) -> str:
    """
    Find the latest checkpoint tag from the checkpoints directory

    Returns:
        Latest checkpoint tag (e.g., "round_05")
    """
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Directory doesn't exist: {checkpoints_dir}")

    # Find all encoder files
    encoder_pattern = os.path.join(checkpoints_dir, 'vae_encoder_*.keras')
    encoder_files = glob.glob(encoder_pattern)

    if not encoder_files:
        raise FileNotFoundError(f"No encoder files found in {checkpoints_dir}")

    # Extract tags and find complete pairs
    valid_tags = []
    for file in encoder_files:
        basename = os.path.basename(file)
        match = re.search(r'vae_encoder_(.+)\.keras', basename)
        if match:
            tag = match.group(1)
            # Verify decoder exists
            decoder_file = os.path.join(checkpoints_dir, f'vae_decoder_{tag}.keras')
            if os.path.exists(decoder_file):
                valid_tags.append(tag)

    if not valid_tags:
        raise FileNotFoundError(f"No valid model pairs found in {checkpoints_dir}")

    # Sort tags to find the latest
    def sort_key(tag_str):
        # Handle final_vX format with highest priority
        if tag_str.startswith("final_"):
            try:
                final_ver = int(tag_str.split('_v')[1])
                return (0, final_ver)
            except:
                return (1, tag_str)
        # Handle round_XX format with secondary priority
        elif tag_str.startswith('round_'):
            try:
                round_num = int(tag_str.split('_')[1])
                return (2, round_num)
            except:
                return (3, tag_str)
        # Handle timestamp format YYYYMMDD_HHMMSS lowest priority
        elif re.match(r'\d{8}_\d{6}', tag_str):
            try:
                timestamp = datetime.strptime(tag_str, '%Y%m%d_%H%M%S')
                return (4, timestamp)
            except:
                return (5, tag_str)
        # Fallback for all other formats
        else:
            return (99, tag_str)

    # Filter for the highest priority group
    priorities = [sort_key(t)[0] for t in valid_tags]
    highest_priority = min(priorities)  # smaller = higher priority

    if highest_priority == 99:
        raise FileNotFoundError(
            f"No valid model tags found (e.g. final_vX, round_XX, YYYYMMDD_HHMMSS)"
        )

    filtered_tags = [t for t in valid_tags if sort_key(t)[0] == highest_priority]

    # Select the latest tag within highest priority group
    filtered_tags.sort(key=sort_key)
    tag = filtered_tags[-1]  # Get the latest
    return tag

class TrainingPipeline:
    """Training pipeline"""
    
    def __init__(self, config, background_data: np.ndarray, strategy=None, start_round=1):
        """
        Initialize training pipeline
        
        Args:
            config: Configuration object
            background_data: Preprocessed background observations
        """
        self.config = config
        self.strategy = strategy or tf.distribute.get_strategy()  # NOTE: is this the source of our setup_gpu_config() issues? 
        
        # Store background data
        self.background_data = background_data.astype(np.float32)
        logger.info(f"Background data shape: {background_data.shape}")
        
        # Initialize components
        self.preprocessor = DataPreprocessor(config)
        self.data_generator = DataGenerator(config, background_data)
        
        # Create VAE model
        with self.strategy.scope():
            self.vae = create_vae_model(config)
        
        self.rf_model = None
        
        # Training history
        self.history = {
            'loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'true_loss': [],
            'false_loss': [],
            'val_loss': [],
            'val_reconstruction_loss': [], 
            'val_kl_loss': [], 
            'val_true_loss': [],
            'val_false_loss': [],
            'learning_rate': []
        }
        
        # Setup directories
        self.setup_directories(start_round)

        # Setup TensorBoard logging
        self.setup_tensorboard_logging(start_round)
        
    def __del__(self):
        """Cleanup TensorBoard writers"""
        if hasattr(self, 'train_writer'):
            self.train_writer.close()
        if hasattr(self, 'val_writer'):
            self.val_writer.close()
    
    def setup_directories(self, start_round=1):
        """Create necessary directories"""
        logger.info("Setting up directories")

        checkpoints_dir = os.path.join(self.config.model_path, 'checkpoints')
        archive_directory(checkpoints_dir, target_dirs=None, round_num=start_round)
        
        plots_dir = os.path.join(self.config.output_path, 'plots')
        archive_directory(plots_dir, target_dirs=None, round_num=start_round)
        
        logger.info(f"Setup directories complete")

    def setup_tensorboard_logging(self, start_round=1):
        """Setup TensorBoard logging"""
        logger.info("Setting up TensorBoard logging")

        logs_dir = os.path.join(self.config.output_path, 'logs')
        archive_directory(logs_dir, target_dirs=['train', 'validation'], round_num=start_round)

        if start_round == 1:
            self.global_step = 0
            logger.info("Starting fresh TensorBoard logs")

        else:
            # NOTE: assuming fixed epochs per round
            self.global_step = (start_round - 1) * self.config.training.epochs_per_round  
            logger.info(f"Resuming TensorBoard logs from step {self.global_step} (round {start_round})")

        # Create TensorBoard writers
        train_log_dir = os.path.join(logs_dir, 'train')
        val_log_dir = os.path.join(logs_dir, 'validation')

        self.train_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_writer = tf.summary.create_file_writer(val_log_dir)

        logger.info(f"TensorBoard logs directory: {logs_dir}")
        logger.info(f"Initial global_step: {self.global_step}")

    def update_learning_rate(self, val_losses):
        """
        Robust adaptive learning rate with multiple safeguards

        Note the following soft constraint: 
        min_learning_rate - base_learning_rate * (1 - reduction_factor) ^ (epochs_per_round / patience_threshold)
          => LR can only reach min_learning_rate during round if above expression is > 0 
          => else LR will reset at start of new round before reaching min_learning_rate
        """

        current_lr = self.vae.optimizer.learning_rate.numpy()
        if current_lr <= self.config.training.min_learning_rate:
            return current_lr
        
        # Use validation loss for better generalization
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
            self.patience_counter = 0
        
        current_val_loss = float(val_losses['total'])
        
        # Check if validation loss improved
        if current_val_loss < self.best_val_loss * (1 - self.config.training.min_pct_improvement):
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Reduce LR if no meaningful improvement for consecutive epochs
        if self.patience_counter >= self.config.training.patience_threshold:
            new_lr = max(current_lr * (1 - self.config.training.reduction_factor), self.config.training.min_learning_rate)
            
            self.vae.optimizer.learning_rate.assign(new_lr)
            self.patience_counter = 0  # Reset counter
            
            logger.info(f"Reduced learning rate: {current_lr:.2e} -> {new_lr:.2e}")
            return new_lr
        
        return current_lr

    def train_round(self, round_idx: int, epochs: int, snr_base: int, snr_range: int):
        """
        Train one round with distributed dataset handling & gradient accumulation
        """
        logger.info(f"Training round {round_idx + 1} - Epochs: {epochs}, SNR: {snr_base}-{snr_base+snr_range}")
        
        # Update generator SNR parameters
        self.config.training.snr_base = snr_base
        self.config.training.snr_range = snr_range
        
        # Use physical batch sizes for memory efficiency
        physical_batch = self.config.training.train_physical_batch_size
        logical_batch = self.config.training.train_logical_batch_size
        accumulation_steps = logical_batch // physical_batch
        val_batch_size = self.config.training.validation_batch_size
        n_samples = self.config.training.num_samples_train
        train_val_split = self.config.training.train_val_split
        
        logger.info(f"Gradient accumulation: {physical_batch} physical, {logical_batch} logical, "
                   f"{accumulation_steps} accumulation steps")
        
        # Generate training data 
        train_data = self.data_generator.generate_training_batch(n_samples * 3)
        
        # Split and trim
        n_train = int(n_samples * 3 * train_val_split)
        n_val = (n_samples * 3) - n_train
        
        n_train_trimmed = (n_train // logical_batch) * logical_batch
        n_val_trimmed = (n_val // val_batch_size) * val_batch_size

        logger.info(f"Data alignment: Train {n_train}→{n_train_trimmed}, Val {n_val}→{n_val_trimmed}")
        
        # Prepare data 
        train_concat = train_data['concatenated'][:n_train_trimmed]
        train_true = train_data['true'][:n_train_trimmed]
        train_false = train_data['false'][:n_train_trimmed]
        
        val_start = n_train
        val_end = val_start + n_val_trimmed
        val_concat = train_data['concatenated'][val_start:val_end]
        val_true = train_data['true'][val_start:val_end]
        val_false = train_data['false'][val_start:val_end]
        
        del train_data
        gc.collect()
        
        # Create datasets with physical batch size
        train_dataset = tf.data.Dataset.from_tensor_slices((
            (train_concat, train_true, train_false), train_concat
        )).shuffle(1000).batch(physical_batch).repeat().prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            (val_concat, val_true, val_false), val_concat  
        )).batch(val_batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        
        # Distribute datasets across GPUs
        train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = self.strategy.experimental_distribute_dataset(val_dataset)
        
        # Training loop with manual forward pass for gradient accumulation
        steps_per_epoch = n_train_trimmed // logical_batch
        val_steps = n_val_trimmed // val_batch_size

        logger.info(f"Infinite dataset setup: {steps_per_epoch} train steps, {val_steps} val steps")
        
        for epoch in range(epochs):
            # Log resources at start of epoch
            logger.info(f"{'-'*30}")
            logger.info(f"Epoch {epoch + 1}/{epochs} Start")
            logger.info(f"{log_system_resources()}")

            # Use gradient accumulation with Keras fit method
            if accumulation_steps > 1:
                # For gradient accumulation, we need custom training loop
                epoch_losses = self._train_epoch_with_accumulation(
                    train_dataset, steps_per_epoch, accumulation_steps
                )
            else:
                # Direct training without accumulation
                epoch_losses = self._train_epoch_direct(
                    train_dataset, steps_per_epoch
                )
            
            # Validation
            val_losses = self._validate_epoch(val_dataset, val_steps)

            # Log results
            logger.info(f"Epoch {epoch + 1} Complete")
            logger.info(f"Train -- Total: {epoch_losses['total']:.4f}, "
                       f"Recon: {epoch_losses['reconstruction']:.4f}, "
                       f"KL: {epoch_losses['kl']:.4f}, "
                       f"True: {epoch_losses['true']:.4f}, "
                       f"False: {epoch_losses['false']:.4f}, ")
            logger.info(f"Val -- Total: {val_losses['total']:.4f}, "
                       f"Recon: {val_losses['reconstruction']:.4f}, "
                       f"KL: {val_losses['kl']:.4f}, "
                       f"True: {val_losses['true']:.4f}, "
                       f"False: {val_losses['false']:.4f}")

            # Update history 
            for key, train_key in [('loss', 'total'), ('reconstruction_loss', 'reconstruction'), 
                                   ('kl_loss', 'kl'), ('true_loss', 'true'), ('false_loss', 'false')]:
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(float(epoch_losses[train_key]))
            for key, val_key in [('val_loss', 'total'), ('val_reconstruction_loss', 'reconstruction'),
                                 ('val_kl_loss', 'kl'), ('val_true_loss', 'true'), ('val_false_loss', 'false')]:
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(float(val_losses[val_key]))
            self.history['learning_rate'].append(float(self.vae.optimizer.learning_rate.numpy()))

            # TensorBoard logging
            with self.train_writer.as_default():
                tf.summary.scalar('total_loss', epoch_losses['total'], step=self.global_step)
                tf.summary.scalar('reconstruction_loss', epoch_losses['reconstruction'], step=self.global_step)
                tf.summary.scalar('kl_loss', epoch_losses['kl'], step=self.global_step)
                tf.summary.scalar('true_loss', epoch_losses['true'], step=self.global_step)
                tf.summary.scalar('false_loss', epoch_losses['false'], step=self.global_step)
                tf.summary.scalar('learning_rate', self.vae.optimizer.learning_rate.numpy(), step=self.global_step)

            with self.val_writer.as_default():
                tf.summary.scalar('validation_total_loss', val_losses['total'], step=self.global_step)
                tf.summary.scalar('validation_reconstruction_loss', val_losses['reconstruction'], step=self.global_step)
                tf.summary.scalar('validation_kl_loss', val_losses['kl'], step=self.global_step)
                tf.summary.scalar('validation_true_loss', val_losses['true'], step=self.global_step)
                tf.summary.scalar('validation_false_loss', val_losses['false'], step=self.global_step)

            # Flush writers to ensure data is written
            self.train_writer.flush()
            self.val_writer.flush()

            # Increment global step
            self.global_step += 1
            
            # Adaptive learning rate
            self.update_learning_rate(val_losses)

            # Log resources at end of epoch  
            logger.info(f"Epoch {epoch + 1}/{epochs} End")
            logger.info(f"{log_system_resources()}")
        
        # Save checkpoint
        self.save_models(
            tag=f"round_{round_idx+1:02d}",
            dir="checkpoints"
        )
        
        # Clean up memory
        del train_concat, train_true, train_false
        del val_concat, val_true, val_false
        gc.collect()

    def _train_epoch_with_accumulation(self, train_dataset, steps_per_epoch, accumulation_steps):
        """Training epoch with gradient accumulation"""
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'true': 0.0,
            'false': 0.0
        }
        train_iterator = iter(train_dataset)
        
        for step in range(steps_per_epoch):
            accumulated_gradients = None
            step_losses = {
                'total': 0.0,
                'reconstruction': 0.0,
                'kl': 0.0,
                'true': 0.0,
                'false': 0.0
            }
            
            # Accumulate gradients over micro-batches
            for micro_step in range(accumulation_steps):
                micro_batch_data = next(train_iterator)
                
                # Use strategy.run to execute train step on all replicas
                per_replica_results = self.strategy.run(self.vae.train_step, args=(micro_batch_data,))
                
                # The train_step already handles gradient computation and application, so we just need to collect the losses
                step_losses['total'] += self.strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, per_replica_results['loss'], axis=None
                )
                step_losses['reconstruction'] += self.strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, per_replica_results['reconstruction_loss'], axis=None
                )
                step_losses['kl'] += self.strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, per_replica_results['kl_loss'], axis=None
                )
                step_losses['true'] += self.strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, per_replica_results['true_loss'], axis=None
                )
                step_losses['false'] += self.strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, per_replica_results['false_loss'], axis=None
                )
            
            # Average losses
            for key in step_losses:
                step_losses[key] /= accumulation_steps
                epoch_losses[key] += step_losses[key]

            logger.info(f"Step {step+1}/{steps_per_epoch}, "
                       f"Total: {step_losses['total']:.4f}, "
                       f"Recon: {step_losses['reconstruction']:.4f}, "
                       f"KL: {step_losses['kl']:.4f}, "
                       f"True: {step_losses['true']:.4f}, "
                       f"False: {step_losses['false']:.4f}")

        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
            
        return epoch_losses

    def _train_epoch_direct(self, train_dataset, steps_per_epoch):
        """Direct training without accumulation"""
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'true': 0.0,
            'false': 0.0
        }
        train_iterator = iter(train_dataset)
        
        for step in range(steps_per_epoch):
            batch_data = next(train_iterator)
            
            # Use strategy.run for distributed execution
            per_replica_results = self.strategy.run(self.vae.train_step, args=(batch_data,))
            
            # Reduce results across replicas
            step_losses = {
                'total': self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results['loss'], axis=None),
                'reconstruction': self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results['reconstruction_loss'], axis=None),
                'kl': self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results['kl_loss'], axis=None),
                'true': self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results['true_loss'], axis=None),
                'false': self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results['false_loss'], axis=None)
            }
            
            # Accumulate losses
            for key in step_losses:
                epoch_losses[key] += step_losses[key]
            
            logger.info(f"Step {step+1}/{steps_per_epoch}, "
                       f"Total: {step_losses['total']:.4f}, "
                       f"Recon: {step_losses['reconstruction']:.4f}, "
                       f"KL: {step_losses['kl']:.4f}, "
                       f"True: {step_losses['true']:.4f}, "
                       f"False: {step_losses['false']:.4f}")
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
            
        return epoch_losses

    def _validate_epoch(self, val_dataset, val_steps):
        """Validation epoch"""
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'true': 0.0,
            'false': 0.0
        }
        val_iterator = iter(val_dataset)
        
        for val_step in range(val_steps):
            val_batch_data = next(val_iterator)
            
            # Use strategy.run for distributed validation
            val_results = self.strategy.run(self.vae.test_step, args=(val_batch_data,))
            
            # Reduce validation results
            val_losses['total'] += self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, val_results['loss'], axis=None
            )
            val_losses['reconstruction'] += self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, val_results['reconstruction_loss'], axis=None
            )
            val_losses['kl'] += self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, val_results['kl_loss'], axis=None
            )
            val_losses['true'] += self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, val_results['true_loss'], axis=None
            )
            val_losses['false'] += self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, val_results['false_loss'], axis=None
            )
        
        # Average validation losses
        for key in val_losses:
            val_losses[key] /= val_steps
            
        return val_losses
    
    def iterative_training(self, start_round=1):
        """
        Perform iterative training with curriculum learning
        """
        n_rounds = self.config.training.num_training_rounds
        epochs = self.config.training.epochs_per_round

        # NOTE: move assertions to main.py
        # if start_round > n_rounds: 
        #     raise MaxRoundsExceededError(
        #         f"Current round ({current_round}) is more than total rounds ({n_rounds})."
        #         "Check model checkpoints, config.py, or CLI args."
        #     )
        
        if start_round > 1:
            logger.info(f"Resuming training from round {start_round}/{n_rounds}")
        else:
            logger.info(f"Starting iterative curriculum training for {n_rounds} rounds")
        
        for round_idx in range(start_round-1, n_rounds):
            snr_base, snr_range = calculate_curriculum_snr(round_idx, n_rounds, self.config.training)

            logger.info(f"\n{'='*50}")
            logger.info(f"ROUND {round_idx + 1}/{n_rounds}")
            logger.info(f"SNR range: {snr_base}-{snr_base+snr_range}")
            logger.info(f"{'='*50}")

            # Reset learning rate & adaptive state before new curriculum stage
            original_lr = self.config.training.base_learning_rate
            current_lr = self.vae.optimizer.learning_rate.numpy()
            self.vae.optimizer.learning_rate.assign(original_lr)
            
            if hasattr(self, 'best_val_loss'):
                delattr(self, 'best_val_loss')
            if hasattr(self, 'patience_counter'):
                delattr(self, 'patience_counter')

            logger.info(f"Curriculum LR reset: {current_lr:.2e} → {original_lr:.2e}")
            
            self.train_round(
                round_idx=round_idx,
                epochs=epochs,
                snr_base=snr_base,
                snr_range=snr_range
            )
            
            # Plot progress every round
            self.plot_training_progress(
                save_path=os.path.join(
                    self.config.output_path,
                    'plots',
                    f'training_progress_round_{round_idx+1}.png'
                )
            )
            
    # NOTE: come back to this later
    def train_random_forest(self):
        """Train Random Forest"""
        logger.info("Training Random Forest classifier...")

        # Load encoder weights if untrained 
        logger.info("Checking if encoder weights appear trained")

        try:
            sample_encoder_weights = self.vae.encoder.layers[1].get_weights()[0]  # First conv layer weights
            mean_absolute_threshold = 0.01

            # Assume trained if mean absolute weights exceed some small threshold
            if np.mean(np.abs(sample_encoder_weights)) > mean_absolute_threshold:   
                logger.info(f"Encoder weights appear trained (>{mean_absolute_threshold})")

            else: 
                logger.info(f"Encoder weights appear random (<{mean_absolute_threshold})")

                try:
                    logger.info(f"Loading pre-trained encoder weights with tag 'final_v1'")
                    self.load_models(tag="final_v1")

                except Exception as e: 
                    logger.warning(f"Failed to load pre-trained encoder weights: {e}")

                    try: 
                        logger.info(f"Loading latest checkpointed weights instead")
                        self.load_models(dir="checkpoints")
                    
                    except Exception as e: 
                        logger.warning(f"Failed to load latest checkpointed weights: {e}")
                        logger.warning("Proceeding with current encoder weights")
        except Exception as e: 
            logger.warning(f"Could not verify encoder weights status: {e}")
            logger.warning(f"Proceeding with current encoder weights")
        
        # Use config values
        n_samples = self.config.training.num_samples_rf // 2  # Split between true/false
        max_chunk_size = self.config.training.max_chunk_size
        
        logger.info(f"Generating {n_samples*2} samples for Random Forest...")
        logger.info(f"Using chunk size: {max_chunk_size}")
        
        try:
            # Generate RF data with config-specified batching
            rf_data = self.data_generator.generate_training_batch(n_samples * 2)
            
            # Separate true and false
            true_data = rf_data['true']
            false_data = rf_data['false']
            
            # Use config values for processing chunks
            chunk_size = min(max_chunk_size, 1000)  # Don't exceed memory limits
            batch_size = min(self.config.training.train_logical_batch_size, 100)  # Small batches for encoder
            
            true_latents = []
            false_latents = []
            
            # Process true data in chunks
            logger.info("Processing true data through encoder...")
            for i in range(0, true_data.shape[0], chunk_size):
                chunk = true_data[i:i+chunk_size]
                chunk_reshaped = chunk.reshape(-1, 16, 
                                             self.config.data.width_bin // self.config.data.downsample_factor, 1)
                _, _, latents = self.vae.encoder.predict(chunk_reshaped, batch_size=batch_size)
                true_latents.append(latents)
                del chunk, chunk_reshaped
                gc.collect()
            
            # Process false data in chunks
            logger.info("Processing false data through encoder...")
            for i in range(0, false_data.shape[0], chunk_size):
                chunk = false_data[i:i+chunk_size]
                chunk_reshaped = chunk.reshape(-1, 16, 
                                             self.config.data.width_bin // self.config.data.downsample_factor, 1)
                _, _, latents = self.vae.encoder.predict(chunk_reshaped, batch_size=batch_size)
                false_latents.append(latents)
                del chunk, chunk_reshaped
                gc.collect()
            
            # Combine latents
            all_true_latents = np.concatenate(true_latents, axis=0)
            all_false_latents = np.concatenate(false_latents, axis=0)
            
            # Clear intermediate data
            del rf_data, true_data, false_data, true_latents, false_latents
            gc.collect()
            
            # Train Random Forest (uses config values internally)
            self.rf_model = RandomForestModel(self.config)
            self.rf_model.train(all_true_latents, all_false_latents)
            
            logger.info("Random Forest training complete")
            
            # Clean up
            del all_true_latents, all_false_latents
            gc.collect()
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            raise
    
    # TODO: add function to plot RF training curves
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot beta-VAE training history"""
        from matplotlib.gridspec import GridSpec
        import matplotlib.lines as mlines
            
        fig = plt.figure(figsize=(25, 12))
        gs = GridSpec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        
        # Top subplot spanning full width - Total Loss
        ax_top = fig.add_subplot(gs[0, :])
        
        # Bottom subplots - Individual losses
        ax_recon = fig.add_subplot(gs[1, 0])
        ax_kl = fig.add_subplot(gs[1, 1]) 
        ax_true = fig.add_subplot(gs[1, 2])
        ax_false = fig.add_subplot(gs[1, 3])
        
        fig.suptitle(f"Beta-VAE Training Progress", fontsize=16)
        
        epochs = range(1, len(self.history.get('loss', [])) + 1)
        
        # Helper function to plot dual y-axis
        def plot_dual_axis(ax, title, train_key, val_key):
            # Create secondary y-axis for learning rate
            ax2 = ax.twinx()
            
            # Plot train and validation on left y-axis
            if train_key in self.history and self.history[train_key]:
                ax.plot(epochs, self.history[train_key], color='blue', label='train', linewidth=2)
            if val_key in self.history and self.history[val_key]:
                ax.plot(epochs, self.history[val_key], color='orange', label='val', linewidth=2)
                
            # Plot learning rate on right y-axis  
            if 'learning_rate' in self.history and self.history['learning_rate']:
                ax2.plot(epochs, self.history['learning_rate'], color='grey', 
                        label='learning rate', linewidth=1, alpha=0.7, linestyle='--')
            
            ax.set_title(title)
            ax.set_xlabel('epoch')
            ax.grid(True, alpha=0.3)
            
            ax2.tick_params(axis='y', labelcolor='grey')
        
        # Top subplot - Total Loss
        plot_dual_axis(ax_top, 'Total Loss', 'loss', 'val_loss')
        
        # Bottom subplots
        plot_dual_axis(ax_recon, 'Reconstruction Loss', 'reconstruction_loss', 'val_reconstruction_loss')
        plot_dual_axis(ax_kl, 'KL Divergence', 'kl_loss', 'val_kl_loss') 
        plot_dual_axis(ax_true, 'True Loss', 'true_loss', 'val_true_loss')
        plot_dual_axis(ax_false, 'False Loss', 'false_loss', 'val_false_loss')

        # Create shared legend at top right of figure
        train_line = mlines.Line2D([], [], color='blue', linewidth=2, label='Train')
        val_line = mlines.Line2D([], [], color='orange', linewidth=2, label='Validation') 
        lr_line = mlines.Line2D([], [], color='grey', linewidth=1, linestyle='--', alpha=0.7, label='Learning Rate')
        
        fig.legend(handles=[train_line, val_line, lr_line], 
                  loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98),
                  frameon=True,
                  fancybox=True,
                  shadow=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(
                os.path.join(self.config.output_path, 'plots', f'training_progress_{timestamp}.png'),
                dpi=300, bbox_inches='tight'
            )
        
        plt.close()
    
    def save_models(self, tag: Optional[str] = None, dir: Optional[str] = None):
        """Save model weights"""
        if tag is None:
            tag = datetime.now().strftime('%Y%m%d_%H%M%S')

        if dir is not None: 
            encoder_path = os.path.join(self.config.model_path, dir, f'vae_encoder_{tag}.keras')
            decoder_path = os.path.join(self.config.model_path, dir, f'vae_decoder_{tag}.keras')
            rf_path = os.path.join(self.config.model_path, dir, f'random_forest_{tag}.joblib')
        else:
            encoder_path = os.path.join(self.config.model_path, f'vae_encoder_{tag}.keras')
            decoder_path = os.path.join(self.config.model_path, f'vae_decoder_{tag}.keras')
            rf_path = os.path.join(self.config.model_path, f'random_forest_{tag}.joblib')

        # Save VAE encoder (main model for inference)
        self.vae.encoder.save(encoder_path)
        logger.info(f"Saved VAE encoder to {encoder_path}")
        
        # Save decoder
        self.vae.decoder.save(decoder_path)
        logger.info(f"Saved VAE decoder to {decoder_path}")
        
        # Save Random Forest
        if self.rf_model is not None:
            self.rf_model.save(rf_path)
            logger.info(f"Saved Random Forest to {rf_path}")

    def load_models(self, tag: Optional[str] = None, dir: Optional[str] = None):
        """Load model weights"""
        import tensorflow as tf

        if tag is None:
            logger.info("No tag specified. Defaulting to 'final'")
            tag = "final"
        original_tag = tag

        # Construct filepaths
        if dir is not None:
            base_dir = os.path.join(self.config.model_path, dir)
            encoder_path = os.path.join(base_dir, f'vae_encoder_{tag}.keras')
            decoder_path = os.path.join(base_dir, f'vae_decoder_{tag}.keras')
            rf_path = os.path.join(base_dir, f'random_forest_{tag}.joblib')
        else:
            base_dir = self.config.model_path
            encoder_path = os.path.join(base_dir, f'vae_encoder_{tag}.keras')
            decoder_path = os.path.join(base_dir, f'vae_decoder_{tag}.keras')
            rf_path = os.path.join(base_dir, f'random_forest_{tag}.joblib')

        # Check if the specified path exists
        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            logger.warning(f"No models tagged as '{original_tag}' in {base_dir}, looking for latest tag instead...")

            tag = get_latest_tag(base_dir)
            logger.info(f"Tag '{original_tag}' not found. Loading latest model with tag: '{tag}'")
            
            # Reconstruct paths with new tag
            encoder_path = os.path.join(base_dir, f'vae_encoder_{tag}.keras')
            decoder_path = os.path.join(base_dir, f'vae_decoder_{tag}.keras')
            rf_path = os.path.join(base_dir, f'random_forest_{tag}.joblib')

        # Load the models
        try:
            if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
                raise FileNotFoundError("Models not found")
            
            logger.info(f"Loading models from {base_dir} with tag '{tag}'")

            # Import Sampling layer for custom_objects (required for model loading)
            from models.vae import Sampling
            
            # Load encoder & decoder
            checkpoint_encoder = tf.keras.models.load_model(
                encoder_path,
                custom_objects={'Sampling': Sampling}
            )
            checkpoint_decoder = tf.keras.models.load_model(
                decoder_path,
                custom_objects={'Sampling': Sampling}
            )
            
            # Transfer weights
            self.vae.encoder.set_weights(checkpoint_encoder.get_weights())
            self.vae.decoder.set_weights(checkpoint_decoder.get_weights())
            
            logger.info("VAE loaded successfully")

            # Load Random Forest if it exists
            if os.path.exists(rf_path):
                # Initialize RF model if it doesn't exist yet
                if self.rf_model is None:
                    from models.random_forest import RandomForestModel
                    self.rf_model = RandomForestModel(self.config)
                
                self.rf_model.load(rf_path)
                logger.info("Random Forest laoded successfully")
            else:
                logger.info(f"Random Forest not found at {rf_path} - this is normal if RF hasn't been trained yet")
            
            logger.info(f"Successfully loaded models from {base_dir} with tag '{tag}'")
            return tag  # Return the actually loaded tag for reference
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

def train_full_pipeline(config, background_data: np.ndarray,
                       strategy=None, tag=None, dir=None, start_round=1) -> TrainingPipeline:
    """
    Train complete SETI ML pipeline
    
    Args:
        config: Configuration object
        background_data: Preprocessed background observations
        strategy: TensorFlow distribution strategy
        
    Returns:
        Trained pipeline object
    """
    # Create pipeline
    pipeline = TrainingPipeline(config, background_data, strategy, start_round)

    # Resume from checkpoint if provided
    if tag:
        logger.info(f"Resuming from checkpoint")
        pipeline.load_models(tag=tag, dir=dir)
    
    # Run iterative training
    pipeline.iterative_training(start_round=start_round)
    
    # Train Random Forest
    pipeline.train_random_forest()
    
    # Save final models
    pipeline.save_models(tag="final_v1")
    
    # Final plot
    pipeline.plot_training_progress(save_path=os.path.join(config.output_path, 'plots', 'training_progress_final_v1.png'))
    
    logger.info("Training complete!")
    
    return pipeline
