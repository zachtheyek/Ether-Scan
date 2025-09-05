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

from preprocessing import DataPreprocessor
from data_generation import DataGenerator
from models.vae import create_vae_model
from models.random_forest import RandomForestModel

logger = logging.getLogger(__name__)

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
        Train one round following author's approach
        
        Args:
            round_idx: Current round index
            epochs: Number of epochs for this round
            snr_base: Base SNR value
            snr_range: SNR range
        """
        logger.info(f"Training round {round_idx} - Epochs: {epochs}, SNR: {snr_base}-{snr_base+snr_range}")
        
        # Update generator SNR parameters
        self.config.training.snr_base = snr_base
        self.config.training.snr_range = snr_range
        
        # Generate training data for this round
        # Author uses large batches: 5000 samples * 3 types = 15000 total
        n_samples = 5000
        
        logger.info(f"Generating {n_samples*3} training samples...")
        
        # Generate balanced dataset
        train_data = self.data_generator.generate_training_batch(n_samples * 3)
        
        # Split into train/validation (80/20)
        n_train = int(n_samples * 3 * 0.8)
        
        train_concat = train_data['concatenated'][:n_train]
        train_true = train_data['true'][:n_train]
        train_false = train_data['false'][:n_train]
        
        val_concat = train_data['concatenated'][n_train:]
        val_true = train_data['true'][n_train:]
        val_false = train_data['false'][n_train:]
        
        # Prepare data in format expected by model
        x_train = (train_concat, train_true, train_false)
        y_train = train_concat
        
        x_val = (val_concat, val_true, val_false)
        y_val = val_concat
        
        # Train with author's batch sizes (1000-2000)
        batch_size = 1000
        
        history = self.vae.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            validation_batch_size=500,
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
            f'vae_round_{round_idx:02d}.h5'
        )
        self.vae.encoder.save(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clean up memory
        del train_data, train_concat, train_true, train_false
        del val_concat, val_true, val_false
        gc.collect()
    
    def iterative_training(self):
        """
        Perform iterative training following author's exact schedule
        Paper shows 20 rounds with varying epochs and consistent SNR
        """
        # Author's training schedule from VAE_NEW_ACCELERATED-BLPC1-8hz-1.py
        epoch_schedule = [150, 150, 150, 150, 150, 150, 150, 100, 100, 100,
                         100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        
        # All rounds use same SNR range: 10-50 (base=10, range=40)
        snr_base = 10
        snr_range = 40
        
        n_rounds = min(len(epoch_schedule), self.config.training.num_training_rounds)
        
        logger.info(f"Starting iterative training for {n_rounds} rounds")
        logger.info(f"SNR range: {snr_base}-{snr_base+snr_range}")
        
        for round_idx in range(n_rounds):
            logger.info(f"\n{'='*50}")
            logger.info(f"ROUND {round_idx + 1}/{n_rounds}")
            logger.info(f"{'='*50}")
            
            self.train_round(
                round_idx=round_idx,
                epochs=epoch_schedule[round_idx],
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
        """
        Train Random Forest after VAE training
        Uses 24,000 samples as per paper
        """
        logger.info("Training Random Forest classifier...")
        
        # Generate RF training data
        n_samples = 12000  # Will generate equal true/false
        
        logger.info(f"Generating {n_samples*2} samples for Random Forest...")
        
        # Generate with final SNR range
        self.config.training.snr_base = 10
        self.config.training.snr_range = 40
        
        rf_data = self.data_generator.generate_training_batch(n_samples * 2)
        
        # Separate true and false
        true_data = rf_data['true']
        false_data = rf_data['false']
        
        # Reshape for encoder
        true_reshaped = true_data.reshape(-1, 16, 512, 1)
        false_reshaped = false_data.reshape(-1, 16, 512, 1)
        
        # Get latent representations
        logger.info("Extracting latent representations...")
        _, _, true_latents = self.vae.encoder.predict(true_reshaped, batch_size=500)
        _, _, false_latents = self.vae.encoder.predict(false_reshaped, batch_size=500)
        
        # Train Random Forest
        self.rf_model = RandomForestModel(self.config)
        self.rf_model.train(true_latents, false_latents)
        
        logger.info("Random Forest training complete")
        
        # Clean up
        del rf_data, true_data, false_data, true_latents, false_latents
        gc.collect()
    
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
