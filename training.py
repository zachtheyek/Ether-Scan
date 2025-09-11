"""
Training pipeline for SETI ML models
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple
import logging
import os
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
                    gpu_info.append(f"GPU{i}: {gpu_util}% util, {mem_used}MB/{mem_total}MB")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        gpu_info = ["GPU info unavailable"]
    
    resource_str = (f"Resources | CPU: {cpu_percent:.1f}% | "
                   f"RAM: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({memory.percent:.1f}%) | "
                   f"{' | '.join(gpu_info)}")
    
    return resource_str

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
    # TODO: add exception handling for all other values of curriculum_schedule
    
    return config.snr_base, int(current_range)

class TrainingPipeline:
    """Training pipeline"""
    
    def __init__(self, config, background_data: np.ndarray, strategy=None):
        """
        Initialize training pipeline
        
        Args:
            config: Configuration object
            background_data: Preprocessed background observations
        """
        self.config = config
        self.strategy = strategy or tf.distribute.get_strategy()
        
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
            'val_loss': []
        }
        
        # Setup directories
        self.setup_directories()

        # Setup TensorBoard logging
        self.log_dir = os.path.join(config.output_path, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create TensorBoard writers
        self.train_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train'))
        self.val_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'validation'))
        
        # Global step counter for TensorBoard
        self.global_step = 0
        
        logger.info(f"TensorBoard logs will be written to: {self.log_dir}")

    def __del__(self):
        """Cleanup TensorBoard writers"""
        if hasattr(self, 'train_writer'):
            self.train_writer.close()
        if hasattr(self, 'val_writer'):
            self.val_writer.close()
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.model_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.model_path, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_path, 'plots'), exist_ok=True)


    def update_learning_rate(self, val_losses, 
                             min_lr_threshold=1e-6, 
                             min_improvement_threshold=0.001, 
                             patience_threshold=5, 
                             reduction_factor=0.2):
        """Robust adaptive learning rate with multiple safeguards"""
        current_lr = self.vae.optimizer.learning_rate.numpy()
        if current_lr <= min_lr_threshold:
            return current_lr
        
        # Use validation loss for better generalization
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
            self.patience_counter = 0
        
        current_val_loss = float(val_losses['total'])
        
        # Check if validation loss improved
        if current_val_loss < self.best_val_loss * (1 - min_improvement_threshold):
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Reduce LR if no improvement for patience_threshold epochs
        if self.patience_counter >= patience_threshold:
            new_lr = max(current_lr * (1 - reduction_factor), min_lr_threshold)
            
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
        
        # Training history tracking
        epoch_metrics = {'loss': [], 'val_loss': []}

        @tf.function
        def compute_gradients_for_accumulation(batch_data, accumulation_steps):
            """Compute gradients for one micro-batch in accumulation"""
            with tf.GradientTape() as tape:
                losses = self.vae.compute_forward_pass(batch_data, training=True)
                # Scale loss by accumulation steps
                scaled_loss = losses['total_loss'] / accumulation_steps
            
            # Compute gradients
            gradients = tape.gradient(scaled_loss, self.vae.trainable_variables)
            return gradients, losses

        for epoch in range(epochs):
            # Log resources at start of epoch
            logger.info(f"{'-'*30}")
            logger.info(f"Epoch {epoch + 1}/{epochs} Start")
            logger.info(f"{log_system_resources()}")
            
            # Training with gradient accumulation
            epoch_losses = {
                'total': 0.0,
                'reconstruction': 0.0,
                'kl': 0.0,
                'true': 0.0,
                'false': 0.0
            }
            train_iterator = iter(train_dataset)
            
            for step in range(steps_per_epoch):
                # Initialize gradient accumulation logic
                accumulated_gradients = []
                step_losses = {
                    'total': 0.0,
                    'reconstruction': 0.0,
                    'kl': 0.0,
                    'true': 0.0,
                    'false': 0.0
                }
                
                # Process accumulation_steps in micro-batches
                for micro_step in range(accumulation_steps):
                    # Get micro-batch and run on all GPUs
                    micro_batch_data = next(train_iterator)
                    per_replica_results = self.strategy.run(
                    compute_gradients_for_accumulation, 
                    args=(micro_batch_data, accumulation_steps)
                    )

                    per_replica_grads, per_replica_losses = per_replica_results[0], per_replica_results[1]
                    
                    # Accumulate gradients
                    if accumulated_gradients is None:
                        accumulated_gradients = per_replica_grads
                    else:
                        accumulated_gradients = [
                            acc_grad + new_grad if acc_grad is not None and new_grad is not None else None
                            for acc_grad, new_grad in zip(accumulated_gradients, per_replica_grads)
                        ]

                    # Accumulate all loss components
                    step_losses['total'] += self.strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, per_replica_losses['total_loss'], axis=None
                    )
                    step_losses['reconstruction'] += self.strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, per_replica_losses['reconstruction_loss'], axis=None
                    )
                    step_losses['kl'] += self.strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, per_replica_losses['kl_loss'], axis=None
                    )
                    step_losses['true'] += self.strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, per_replica_losses['true_loss'], axis=None
                    )
                    step_losses['false'] += self.strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, per_replica_losses['false_loss'], axis=None
                    )
                
                # Apply accumulated gradients
                if accumulated_gradients is not None:
                    valid_grads_and_vars = [
                        (grad, var) for grad, var in zip(accumulated_gradients, self.vae.trainable_variables)
                        if grad is not None
                    ]
            
                    if valid_grads_and_vars:
                        self.vae.optimizer.apply_gradients(valid_grads_and_vars)
                
                # Average step losses by accumulation steps
                for key in step_losses:
                    step_losses[key] /= accumulation_steps
                    epoch_losses[key] += step_losses[key]
                
                if step % 10 == 0:
                    logger.info(f"Step {step+1}/{steps_per_epoch}, "
                               f"Total: {step_losses['total']:.4f}, "
                               f"Recon: {step_losses['reconstruction']:.4f}, "
                               f"KL: {step_losses['kl']:.4f}, "
                               f"True: {step_losses['true']:.4f}, "
                               f"False: {step_losses['false']:.4f}")
            
            # Validation (can use larger batches - no gradients computed)
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
                val_results = self.strategy.run(self.vae.test_step, args=(val_batch_data,))
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
            
            # Log epoch results
            for key in epoch_losses:
                epoch_losses[key] /= steps_per_epoch
            for key in val_losses:
                val_losses[key] /= val_steps

            epoch_metrics['loss'].append(float(epoch_losses['total']))
            epoch_metrics['val_loss'].append(float(val_losses['total']))
            
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

            # Add TensorBoard logging
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
        
        # Update history 
        for key, values in epoch_metrics.items():
            if key in self.history:
                self.history[key].extend(values)
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.model_path, 'checkpoints', f'vae_round_{round_idx+1:02d}.h5'
        )
        self.vae.encoder.save(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clean up memory
        del train_concat, train_true, train_false
        del val_concat, val_true, val_false
        gc.collect()
    
    def iterative_training(self):
        """
        Perform iterative training with curriculum learning
        """
        epochs = self.config.training.epochs_per_round
        n_rounds = self.config.training.num_training_rounds
        
        logger.info(f"Starting iterative curriculum training for {n_rounds} rounds")
        
        for round_idx in range(n_rounds):
            snr_base, snr_range = calculate_curriculum_snr(round_idx, n_rounds, self.config.training)

            logger.info(f"\n{'='*50}")
            logger.info(f"ROUND {round_idx + 1}/{n_rounds}")
            logger.info(f"SNR range: {snr_base}-{snr_base+snr_range}")
            logger.info(f"{'='*50}")
            
            self.train_round(
                round_idx=round_idx,
                epochs=epochs,
                snr_base=snr_base,
                snr_range=snr_range
            )
            
            # Plot progress every 5 rounds
            if (round_idx + 1) % 5 == 0:
                self.plot_training_history(
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
        
        # Use config values
        n_samples = self.config.training.num_samples_rf // 2  # Split between true/false
        max_chunk_size = getattr(self.config.training, 'max_chunk_size', 1000)
        
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
            batch_size = min(self.config.training.batch_size, 100)  # Small batches for encoder
            
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
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history matching author's style"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        fig.suptitle(f"Model Training Progress", fontsize=16)
        
        # False/True clustering losses
        if 'false_loss' in self.history and self.history['false_loss']:
            ax1.plot(self.history['false_loss'], label='false_loss')
        if 'true_loss' in self.history and self.history['true_loss']:
            ax1.plot(self.history['true_loss'], label='true_loss')
        ax1.set_title('Model Clustering Loss')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.legend()
        ax1.grid()
        
        # Total loss
        if 'loss' in self.history and self.history['loss']:
            ax2.plot(self.history['loss'], label='train')
        if 'val_loss' in self.history and self.history['val_loss']:
            ax2.plot(self.history['val_loss'], label='validation')
        ax2.set_title('Total Loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend()
        ax2.grid()
        
        # Reconstruction loss
        if 'reconstruction_loss' in self.history and self.history['reconstruction_loss']:
            ax3.plot(self.history['reconstruction_loss'])
        ax3.set_title('Model Reconstruction')
        ax3.set_ylabel('loss')
        ax3.set_xlabel('epoch')
        ax3.grid()
        
        # KL divergence
        if 'kl_loss' in self.history and self.history['kl_loss']:
            ax4.plot(self.history['kl_loss'])
        ax4.set_title('Model Divergence')
        ax4.set_ylabel('loss')
        ax4.set_xlabel('epoch')
        ax4.grid()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(
                os.path.join(self.config.output_path, 'plots', f'training_history_{timestamp}.png')
            )
        
        plt.close()
    
    def save_models(self, tag: Optional[str] = None):
        """Save trained models"""
        if tag is None:
            tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save VAE encoder (main model for inference)
        encoder_path = os.path.join(self.config.model_path, f'vae_encoder_{tag}.h5')
        self.vae.encoder.save(encoder_path)
        logger.info(f"Saved VAE encoder to {encoder_path}")
        
        # Save decoder
        decoder_path = os.path.join(self.config.model_path, f'vae_decoder_{tag}.h5')
        self.vae.decoder.save(decoder_path)
        logger.info(f"Saved VAE decoder to {decoder_path}")
        
        # Save Random Forest
        if self.rf_model is not None:
            rf_path = os.path.join(self.config.model_path, f'random_forest_{tag}.joblib')
            self.rf_model.save(rf_path)
            logger.info(f"Saved Random Forest to {rf_path}")

def train_full_pipeline(config, background_data: np.ndarray,
                       strategy=None) -> TrainingPipeline:
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
    pipeline = TrainingPipeline(config, background_data, strategy)
    
    # Run iterative training
    pipeline.iterative_training()
    
    # Train Random Forest
    pipeline.train_random_forest()
    
    # Save final models
    pipeline.save_models("final")
    
    # Final plot
    pipeline.plot_training_history()
    
    logger.info("Training complete!")
    
    return pipeline
