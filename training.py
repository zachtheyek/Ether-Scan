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

from .preprocessing import DataPreprocessor
from .data_generation import DataGenerator, create_mixed_training_batch
from .models.vae import create_vae_model
from .models.random_forest import train_random_forest

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Main training pipeline for SETI models"""
    
    def __init__(self, config, background_data: np.ndarray):
        """
        Initialize training pipeline
        
        Args:
            config: Configuration object
            background_data: Array of background observations for data generation
        """
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.data_generator = DataGenerator(config, background_data)
        
        # Initialize models
        self.vae = create_vae_model(config)
        self.rf_model = None
        
        # Training history
        self.history = {
            'loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'clustering_loss': [],
            'val_loss': []
        }
        
        # Setup directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.model_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.model_path, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_path, 'plots'), exist_ok=True)
        
    def prepare_training_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare training and validation datasets
        
        Returns:
            Training and validation tf.data.Dataset objects
        """
        logger.info("Generating training data...")
        
        # Generate datasets
        train_data = self.data_generator.generate_training_set()
        val_data = self.data_generator.generate_test_set()
        
        # Preprocess
        logger.info("Preprocessing data...")
        
        # Training dataset
        train_concat = self.preprocessor.downsample_frequency(train_data['concatenated'])
        train_true = self.preprocessor.downsample_frequency(train_data['true_combined'])
        train_false = self.preprocessor.downsample_frequency(train_data['false'])
        
        # Prepare for model
        train_concat_flat = self.preprocessor.prepare_batch(train_concat)
        
        # Create TF dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((
            (train_concat_flat, train_true, train_false),
            train_concat_flat  # Target is same as input for autoencoder
        ))
        
        train_dataset = train_dataset.batch(self.config.training.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset
        val_concat = self.preprocessor.downsample_frequency(val_data['true'])
        val_true = self.preprocessor.downsample_frequency(val_data['true'])
        val_false = self.preprocessor.downsample_frequency(val_data['false'])
        
        val_concat_flat = self.preprocessor.prepare_batch(val_concat)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            (val_concat_flat, val_true, val_false),
            val_concat_flat
        ))
        
        val_dataset = val_dataset.batch(self.config.training.validation_batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def train_vae(self, epochs: Optional[int] = None, 
                 save_checkpoints: bool = True) -> Dict:
        """
        Train the VAE model
        
        Args:
            epochs: Number of epochs (uses config if None)
            save_checkpoints: Whether to save checkpoints
            
        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.training.epochs_per_round
            
        logger.info(f"Training VAE for {epochs} epochs...")
        
        # Prepare data
        train_dataset, val_dataset = self.prepare_training_data()
        
        # Callbacks
        callbacks = []
        
        if save_checkpoints:
            checkpoint_path = os.path.join(
                self.config.model_path, 
                'checkpoints',
                'vae_checkpoint_{epoch:02d}.h5'
            )
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    save_weights_only=True,
                    save_freq='epoch'
                )
            )
        
        # Train
        history = self.vae.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Update history
        for key in self.history:
            if key in history.history:
                self.history[key].extend(history.history[key])
        
        return history.history
    
    def train_random_forest(self) -> None:
        """Train Random Forest classifier using trained VAE encoder"""
        logger.info("Training Random Forest classifier...")
        
        # Generate training data for RF
        rf_train_data = {
            'true': self.preprocessor.prepare_batch(
                self.preprocessor.downsample_frequency(
                    self.data_generator.generate_batch(4000, "true")
                )
            ),
            'false': self.preprocessor.prepare_batch(
                self.preprocessor.downsample_frequency(
                    self.data_generator.generate_batch(4000, "false")
                )
            )
        }
        
        # Train RF
        self.rf_model = train_random_forest(
            self.vae.encoder,
            rf_train_data,
            self.config
        )
        
    def iterative_training(self, n_rounds: Optional[int] = None) -> None:
        """
        Perform iterative training with varying SNR
        
        Args:
            n_rounds: Number of training rounds
        """
        if n_rounds is None:
            n_rounds = self.config.training.num_training_rounds
            
        logger.info(f"Starting iterative training for {n_rounds} rounds...")
        
        for round_idx in range(n_rounds):
            logger.info(f"Training round {round_idx + 1}/{n_rounds}")
            
            # Vary SNR for each round
            original_snr = self.config.training.snr_base
            self.config.training.snr_base = original_snr + (round_idx * 2)
            
            # Train VAE
            self.train_vae(epochs=self.config.training.epochs_per_round)
            
            # Save intermediate model
            self.save_vae_checkpoint(f"round_{round_idx}")
            
            # Reset SNR
            self.config.training.snr_base = original_snr
            
        # Train Random Forest after VAE training
        self.train_random_forest()
        
    def evaluate(self, test_data: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """
        Evaluate trained models
        
        Args:
            test_data: Optional test data dictionary
            
        Returns:
            Evaluation metrics
        """
        if test_data is None:
            test_data = self.data_generator.generate_test_set()
            
        metrics = {}
        
        # Preprocess test data
        test_true = self.preprocessor.prepare_batch(
            self.preprocessor.downsample_frequency(test_data['true'])
        )
        test_false = self.preprocessor.prepare_batch(
            self.preprocessor.downsample_frequency(test_data['false'])
        )
        
        # VAE reconstruction metrics
        true_recon = self.vae.predict(test_true, batch_size=self.config.inference.batch_size)
        false_recon = self.vae.predict(test_false, batch_size=self.config.inference.batch_size)
        
        metrics['vae_reconstruction_error_true'] = np.mean((test_true - true_recon) ** 2)
        metrics['vae_reconstruction_error_false'] = np.mean((test_false - false_recon) ** 2)
        
        # Random Forest metrics
        if self.rf_model is not None:
            # Get latent representations
            true_latents = self.vae.encoder.predict(test_true)[2]
            false_latents = self.vae.encoder.predict(test_false)[2]
            
            # Predict
            true_preds = self.rf_model.predict(true_latents)
            false_preds = self.rf_model.predict(false_latents)
            
            # Calculate metrics
            metrics['rf_true_positive_rate'] = np.mean(true_preds == 1)
            metrics['rf_false_positive_rate'] = np.mean(false_preds == 1)
            metrics['rf_accuracy'] = np.mean(
                np.concatenate([true_preds == 1, false_preds == 0])
            )
            
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        ax1.plot(self.history['loss'], label='Train')
        if 'val_loss' in self.history:
            ax1.plot(self.history['val_loss'], label='Validation')
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Reconstruction loss
        ax2.plot(self.history['reconstruction_loss'])
        ax2.set_title('Reconstruction Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # KL loss
        ax3.plot(self.history['kl_loss'])
        ax3.set_title('KL Divergence Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # Clustering loss
        ax4.plot(self.history['clustering_loss'])
        ax4.set_title('Clustering Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        
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
            
        # Save VAE
        vae_path = os.path.join(self.config.model_path, f'vae_encoder_{tag}.h5')
        self.vae.encoder.save(vae_path)
        logger.info(f"Saved VAE encoder to {vae_path}")
        
        # Save decoder
        decoder_path = os.path.join(self.config.model_path, f'vae_decoder_{tag}.h5')
        self.vae.decoder.save(decoder_path)
        logger.info(f"Saved VAE decoder to {decoder_path}")
        
        # Save RF
        if self.rf_model is not None:
            rf_path = os.path.join(self.config.model_path, f'random_forest_{tag}.joblib')
            self.rf_model.save(rf_path)
    
    def save_vae_checkpoint(self, tag: str):
        """Save VAE checkpoint"""
        checkpoint_path = os.path.join(
            self.config.model_path,
            'checkpoints',
            f'vae_{tag}.h5'
        )
        self.vae.encoder.save(checkpoint_path)
        logger.info(f"Saved VAE checkpoint to {checkpoint_path}")

def train_full_pipeline(config, background_data: np.ndarray,
                       n_rounds: int = 20) -> TrainingPipeline:
    """
    Train the complete SETI ML pipeline
    
    Args:
        config: Configuration object
        background_data: Background observations for data generation
        n_rounds: Number of training rounds
        
    Returns:
        Trained pipeline object
    """
    # Create pipeline
    pipeline = TrainingPipeline(config, background_data)
    
    # Train
    pipeline.iterative_training(n_rounds)
    
    # Evaluate
    metrics = pipeline.evaluate()
    logger.info(f"Final evaluation metrics: {metrics}")
    
    # Save results
    pipeline.plot_training_history()
    pipeline.save_models()
    
    return pipeline
