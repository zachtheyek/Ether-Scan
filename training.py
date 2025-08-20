"""
Training pipeline for SETI ML models
Fixed to match paper's training methodology
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

from preprocessing import DataPreprocessor
from data_generation import DataGenerator, create_mixed_training_batch
from models.vae import create_vae_model
from models.random_forest import train_random_forest, RandomForestModel

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Main training pipeline for SETI models"""
    
    def __init__(self, config, background_data: np.ndarray):
        """
        Initialize training pipeline
        
        Args:
            config: Configuration object
            background_data: Array of background observations for data generation
                           Expected shape: (n_backgrounds, 6, 16, 512) after preprocessing
        """
        logger.info("Initializing TrainingPipeline...")
        self.config = config
        
        logger.info("Creating DataPreprocessor...")
        self.preprocessor = DataPreprocessor(config)
        
        logger.info(f"Background data shape: {background_data.shape}")
        logger.info("Creating DataGenerator...")
        self.data_generator = DataGenerator(config, background_data)
        
        # Initialize models
        logger.info("Creating VAE model...")
        self.vae = create_vae_model(config)
        logger.info("VAE model created successfully")
        
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
        Paper: 120,000 samples for training, 24,000 for validation
        
        Returns:
            Training and validation tf.data.Dataset objects
        """
        logger.info("Preparing training data...")
        
        # Generate full training set as per paper
        train_data = self.data_generator.generate_training_set()
        val_data = self.data_generator.generate_test_set()
        
        # Create TF datasets
        def create_dataset(data_dict, shuffle=True):
            """Helper to create TF dataset from dictionary"""
            concatenated = data_dict.get('concatenated', data_dict.get('true'))
            true_data = data_dict['true']
            false_data = data_dict['false']
            
            # Prepare for model input (add channel dimension and flatten batch)
            concatenated_flat = self.preprocessor.prepare_batch(concatenated)
            
            # Create dataset
            dataset = tf.data.Dataset.from_tensor_slices((
                (concatenated_flat, true_data, false_data),
                concatenated_flat  # Target is same as input for autoencoder
            ))
            
            if shuffle:
                dataset = dataset.shuffle(buffer_size=10000)
            
            return dataset
        
        train_dataset = create_dataset(train_data, shuffle=True)
        val_dataset = create_dataset(val_data, shuffle=False)
        
        # Batch and prefetch
        train_dataset = train_dataset.batch(self.config.training.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        val_dataset = val_dataset.batch(self.config.training.validation_batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def train_vae(self, epochs: Optional[int] = None, 
                 save_checkpoints: bool = True) -> Dict:
        """
        Train the VAE model
        Paper: 100 epochs per round, 20 rounds total
        
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
                'vae_checkpoint_{epoch:02d}.weights.h5'
            )
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    save_weights_only=True,
                    save_freq='epoch',
                    verbose=1
                )
            )
            
        # Early stopping to prevent overfitting
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
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
        """
        Train Random Forest classifier using trained VAE encoder
        Paper: 1000 estimators, trained on latent representations
        """
        logger.info("Training Random Forest classifier...")
        
        # Generate training data for RF (24,000 samples as per paper)
        n_samples = 24000 // 2  # Half true, half false
        
        true_data = self.data_generator.generate_batch(n_samples, "true")
        false_data = self.data_generator.generate_batch(n_samples, "false")
        
        # Prepare for VAE input
        true_prepared = self.preprocessor.prepare_batch(true_data)
        false_prepared = self.preprocessor.prepare_batch(false_data)
        
        # Get latent representations
        logger.info("Extracting latent representations...")
        _, _, true_latents = self.vae.encoder.predict(true_prepared, batch_size=64)
        _, _, false_latents = self.vae.encoder.predict(false_prepared, batch_size=64)
        
        # Create and train RF model
        self.rf_model = RandomForestModel(self.config)
        self.rf_model.train(true_latents, false_latents)
        
        logger.info("Random Forest training complete")
        
    def iterative_training(self, n_rounds: Optional[int] = None) -> None:
        """
        Perform iterative training with varying SNR
        Paper: 20 rounds, varying SNR from 10 to 50
        
        Args:
            n_rounds: Number of training rounds (default: 20)
        """
        if n_rounds is None:
            n_rounds = self.config.training.num_training_rounds
            
        logger.info(f"Starting iterative training for {n_rounds} rounds...")
        
        for round_idx in range(n_rounds):
            logger.info(f"\n{'='*50}")
            logger.info(f"Training round {round_idx + 1}/{n_rounds}")
            logger.info(f"{'='*50}")
            
            # Vary SNR for each round (gradually increase difficulty)
            # Start with easier signals (higher SNR) and progress to harder ones
            snr_progression = 50 - (round_idx * 2)  # 50, 48, 46, ... 12, 10
            self.config.training.snr_base = max(10, snr_progression)
            
            logger.info(f"SNR base for this round: {self.config.training.snr_base}")
            
            # Train VAE
            self.train_vae(epochs=self.config.training.epochs_per_round)
            
            # Save intermediate model
            self.save_vae_checkpoint(f"round_{round_idx:02d}")
            
        # Train Random Forest after VAE training is complete
        logger.info("\nTraining Random Forest on final VAE encoder...")
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
        
        # Prepare test data
        test_true = self.preprocessor.prepare_batch(test_data['true'])
        test_false = self.preprocessor.prepare_batch(test_data['false'])
        
        # VAE reconstruction metrics
        true_recon = self.vae.predict(test_true, batch_size=64)
        false_recon = self.vae.predict(test_false, batch_size=64)
        
        metrics['vae_reconstruction_error_true'] = np.mean((test_true - true_recon) ** 2)
        metrics['vae_reconstruction_error_false'] = np.mean((test_false - false_recon) ** 2)
        
        # Random Forest metrics
        if self.rf_model is not None:
            # Get latent representations
            _, _, true_latents = self.vae.encoder.predict(test_true, batch_size=64)
            _, _, false_latents = self.vae.encoder.predict(test_false, batch_size=64)
            
            # Predict
            true_preds = self.rf_model.predict(true_latents)
            false_preds = self.rf_model.predict(false_latents)
            
            # Calculate metrics
            metrics['rf_true_positive_rate'] = np.mean(true_preds == 1)
            metrics['rf_false_positive_rate'] = np.mean(false_preds == 1)
            metrics['rf_accuracy'] = np.mean(
                np.concatenate([true_preds == 1, false_preds == 0])
            )
            
            logger.info(f"Evaluation metrics: {metrics}")
            
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        if self.history['loss']:
            ax1.plot(self.history['loss'], label='Train')
            if 'val_loss' in self.history and self.history['val_loss']:
                ax1.plot(self.history['val_loss'], label='Validation')
            ax1.set_title('Total Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
        
        # Reconstruction loss
        if self.history['reconstruction_loss']:
            ax2.plot(self.history['reconstruction_loss'])
            ax2.set_title('Reconstruction Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        # KL loss
        if self.history['kl_loss']:
            ax3.plot(self.history['kl_loss'])
            ax3.set_title('KL Divergence Loss')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.grid(True)
        
        # Clustering loss
        if self.history['clustering_loss']:
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
            
        # Save VAE encoder
        encoder_path = os.path.join(self.config.model_path, f'vae_encoder_{tag}.h5')
        self.vae.encoder.save(encoder_path)
        logger.info(f"Saved VAE encoder to {encoder_path}")
        
        # Save VAE decoder
        decoder_path = os.path.join(self.config.model_path, f'vae_decoder_{tag}.h5')
        self.vae.decoder.save(decoder_path)
        logger.info(f"Saved VAE decoder to {decoder_path}")
        
        # Save RF
        if self.rf_model is not None:
            rf_path = os.path.join(self.config.model_path, f'random_forest_{tag}.joblib')
            self.rf_model.save(rf_path)
            logger.info(f"Saved Random Forest to {rf_path}")
    
    def save_vae_checkpoint(self, tag: str):
        """Save VAE checkpoint"""
        checkpoint_path = os.path.join(
            self.config.model_path,
            'checkpoints',
            f'vae_{tag}.weights.h5'
        )
        self.vae.save_weights(checkpoint_path)
        logger.info(f"Saved VAE checkpoint to {checkpoint_path}")

def train_full_pipeline(config, background_data: np.ndarray,
                       n_rounds: int = 20) -> TrainingPipeline:
    """
    Train the complete SETI ML pipeline
    Paper: 20 rounds of training with 100 epochs each
    
    Args:
        config: Configuration object
        background_data: Background observations for data generation
        n_rounds: Number of training rounds (default: 20)
        
    Returns:
        Trained pipeline object
    """
    # Create pipeline
    logger.info("Creating TrainingPipeline...")
    pipeline = TrainingPipeline(config, background_data)
    logger.info("TrainingPipeline created successfully")
    
    # Train iteratively
    logger.info(f"Starting iterative training for {n_rounds} rounds...")
    pipeline.iterative_training(n_rounds)
    logger.info("Training completed")
    
    # Evaluate
    metrics = pipeline.evaluate()
    logger.info(f"Final evaluation metrics: {metrics}")
    
    # Save results
    pipeline.plot_training_history()
    pipeline.save_models("final")
    
    return pipeline
