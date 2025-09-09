"""
Training pipeline for SETI ML models
Fixed to match paper's training methodology exactly
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import gc

from config import TrainingConfig
from preprocessing import DataPreprocessor
from data_generation import DataGenerator
from models.vae import create_vae_model
from models.random_forest import RandomForestModel

logger = logging.getLogger(__name__)

def cleanup_memory():
    """Force memory cleanup for distributed training"""
    import gc
    gc.collect()
    
    # Clear TensorFlow's session state
    try:
        tf.keras.backend.clear_session()
    except:
        pass
    
    # Force GPU memory cleanup if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.reset_memory_growth(gpu)
        except:
            pass

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
        # Exponential decay - stay easy longer, then get hard quickly
        current_range = config.final_snr_range + (config.initial_snr_range - config.final_snr_range) * np.exp(config.exponential_decay_rate * progress)
    elif config.curriculum_schedule == "step":
        # Step function - easy for first part, hard for second part
        if round_idx < config.easy_rounds:
            current_range = config.initial_snr_range
        else:
            current_range = config.final_snr_range
        # NOTE: add mechanism for more step changes
    # NOTE: add exception for all other values of curriculum_schedule
    
    return config.snr_base, int(current_range)

class TrainingPipeline:
    """Training pipeline matching author's methodology"""
    
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
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.model_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.model_path, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_path, 'plots'), exist_ok=True)

    def train_round(self, round_idx: int, epochs: int, snr_base: int, snr_range: int):
        """
        Train one round using config-specified parameters & proper distributed dataset handling
        """
        logger.info(f"Training round {round_idx + 1} - Epochs: {epochs}, SNR: {snr_base}-{snr_base+snr_range}")
        
        # Update generator SNR parameters
        self.config.training.snr_base = snr_base
        self.config.training.snr_range = snr_range
        
        # Use config values for sample count and batch size
        batch_size = self.config.training.batch_size
        val_batch_size = self.config.training.validation_batch_size
        n_samples = self.config.training.num_samples_train
        train_val_split = self.config.training.train_val_split
        
        logger.info(f"Using config values - Samples: {n_samples}, Batch size: {batch_size}")
        
        # Generate training data (will use config-specified chunking)
        train_data = self.data_generator.generate_training_batch(n_samples * 3)
        
        # Split into train/validation (80/20)
        n_train = int(n_samples * 3 * train_val_split)
        n_val = (n_samples * 3) - n_train

        # Calculate trimming point so samples are divisible by batch size (to avoid OUT_OF_RANGE)
        n_train_trimmed = (n_train // batch_size) * batch_size
        n_val_trimmed = (n_val // val_batch_size) * val_batch_size

        logger.info(f"Data alignment: Train {n_train}→{n_train_trimmed}, Val {n_val}→{n_val_trimmed}")
        logger.info(f"Steps per epoch: Train {n_train_trimmed // batch_size}, Val {n_val_trimmed // val_batch_size}")

        # Split and trim in one operation to avoid intermediate arrays
        train_concat = train_data['concatenated'][:n_train_trimmed]
        train_true = train_data['true'][:n_train_trimmed]
        train_false = train_data['false'][:n_train_trimmed]
     
        val_start = n_train  # Start from original split point
        val_end = val_start + n_val_trimmed  # Take only trimmed amount
        val_concat = train_data['concatenated'][val_start:val_end]
        val_true = train_data['true'][val_start:val_end]
        val_false = train_data['false'][val_start:val_end]
               
        # Clear original data to save memory
        del train_data
        gc.collect()
        
        # Prepare data in format expected by model
        x_train = (train_concat, train_true, train_false)
        y_train = train_concat
        
        x_val = (val_concat, val_true, val_false)
        y_val = val_concat

       # Add custom callback for memory cleanup
        class MemoryCleanupCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                cleanup_memory()
        
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),  # Terminate training if loss goes to NaN/Inf
            MemoryCleanupCallback(),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
            )  # Reduce learning rate on loss plateau to prevent instability
        ]
        
        logger.info(f"Training with batch_size={batch_size}, val_batch_size={val_batch_size}")

        history = self.vae.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,  # Use config value
            epochs=epochs,
            validation_data=(x_val, y_val),
            validation_batch_size=val_batch_size,  # Use config value
            callbacks=callbacks,
            verbose=1
        )
        
        # Update history
        for key in history.history:
            if key in self.history:
                self.history[key].extend(history.history[key])
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.model_path,
            'checkpoints',
            f'vae_round_{round_idx+1:02d}.h5'
        )
        self.vae.encoder.save(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clean up memory
        del train_concat, train_true, train_false
        del val_concat, val_true, val_false
        del x_train, y_train, x_val, y_val
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
    
    def train_random_forest(self):
        """Train Random Forest using config-specified parameters"""
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
                       n_rounds: int = 20, strategy=None) -> TrainingPipeline:
    """
    Train complete SETI ML pipeline following author's methodology
    
    Args:
        config: Configuration object
        background_data: Preprocessed background observations
        n_rounds: Number of training rounds (default: 20 as per paper)
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
