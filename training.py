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
    
    def __init__(self, config, background_data: np.ndarray, strategy=None):
        """
        Initialize training pipeline with distributed strategy support
        
        Args:
            config: Configuration object
            background_data: Array of background observations for data generation
                           Expected shape: (n_backgrounds, 6, 16, 512) after preprocessing
        """
        logger.info("Initializing TrainingPipeline...")
        self.config = config

        self.strategy = strategy or tf.distribute.get_strategy()
        logger.info(f"Using strategy: {self.strategy.__class__.__name__} with {self.strategy.num_replicas_in_sync} replicas")

        # Convert background data to float32
        logger.info("Converting background data to float32...")
        self.background_data = background_data.astype(np.float32)
        
        logger.info("Creating DataPreprocessor...")
        self.preprocessor = DataPreprocessor(config)
        
        logger.info(f"Background data shape: {background_data.shape}")
        logger.info("Creating DataGenerator...")
        self.data_generator = DataGenerator(config, background_data)
        
        # Create models within strategy scope for distributed training
        with self.strategy.scope():
            logger.info("Creating VAE model within distributed scope...")
            self.vae = create_vae_model(config)
            
            # Scale learning rate by number of replicas
            scaled_lr = config.model.learning_rate * self.strategy.num_replicas_in_sync
            logger.info(f"Scaling learning rate from {config.model.learning_rate} to {scaled_lr}")
            
            # Recompile with scaled learning rate
            self.vae.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=scaled_lr)
            )
            logger.info("VAE model created and compiled for distributed training")
        
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
        Prepare training and validation datasets using memory-efficient generators
        
        Returns:
            Training and validation tf.data.Dataset objects
        """
        logger.info("Preparing training data with memory-efficient generators...")

        # Calculate steps per epoch
        steps_per_epoch = self.config.training.num_samples_train // self.config.training.batch_size
        val_steps = self.config.training.num_samples_test // self.config.training.validation_batch_size
        
        def train_generator():
            """Generator that yields single samples for training"""
            while True:
                # Generate a small batch to work with
                batch_size = self.config.training.samples_per_generator_call
                
                # Mix the data (1/4 none, 1/4 true, 1/4 false, 1/4 mixed)
                quarter_size = batch_size // 4
                
                none_data = self.data_generator.generate_batch(quarter_size, "none")
                true_data = self.data_generator.generate_batch(quarter_size, "true")
                false_data = self.data_generator.generate_batch(quarter_size, "false")
                
                # For mixed data, we need to add RFI to true signals
                mixed_data = self.data_generator.generate_batch(quarter_size, "true")
                
                # Add RFI signals to the mixed_data
                for i in range(quarter_size):
                    for obs_idx in range(6):  # Add RFI to all 6 observations
                        # Generate RFI parameters
                        rfi_snr = np.random.uniform(10, 30)
                        rfi_drift_rate = np.random.uniform(-5, 5)
                        
                        from data_generation import inject_signal
                        mixed_data[i, obs_idx], _, _ = inject_signal(
                            mixed_data[i, obs_idx], 
                            rfi_snr, 
                            rfi_drift_rate
                        )
                
                # Combine all data types
                mixed_batch = np.concatenate([none_data, true_data, false_data, mixed_data], axis=0)
                
                # Yield individual cadence samples in the correct format
                for i in range(len(mixed_batch)):
                    # Get the cadence data for this sample: (6, 16, 512)
                    cadence_data = mixed_batch[i]  # Shape: (6, 16, 512)
                    
                    # Add channel dimension: (6, 16, 512, 1)
                    cadence_with_channels = np.expand_dims(cadence_data, axis=-1)
                    
                    # Simple input/target format - model will handle reshaping
                    yield (cadence_with_channels, cadence_with_channels)
        
        def val_generator():
            """Generator for validation data"""
            while True:
                # Similar to train_generator but simpler
                batch_size = self.config.training.samples_per_generator_call
                val_data = self.data_generator.generate_batch(batch_size, "true")
                
                for i in range(len(val_data)):
                    # Get the cadence data for this sample: (6, 16, 512)
                    cadence_data = val_data[i]  # Shape: (6, 16, 512)
                    
                    # Add channel dimension: (6, 16, 512, 1)
                    cadence_with_channels = np.expand_dims(cadence_data, axis=-1)
                    
                    # Simple input/target format for validation
                    yield (cadence_with_channels, cadence_with_channels)
        
        # Define output signature for the generators - simplified
        output_signature = (
            tf.TensorSpec(shape=(6, 16, 512, 1), dtype=tf.float32),  # input
            tf.TensorSpec(shape=(6, 16, 512, 1), dtype=tf.float32)   # target
        )
        
        # Create datasets from generators
        train_dataset = tf.data.Dataset.from_generator(
            train_generator,
            output_signature=output_signature
        )
        
        val_dataset = tf.data.Dataset.from_generator(
            val_generator,
            output_signature=output_signature
        )
        
        # Batch and optimize
        train_dataset = train_dataset.batch(self.config.training.batch_size)
        train_dataset = train_dataset.prefetch(self.config.training.prefetch_buffer)
        
        val_dataset = val_dataset.batch(self.config.training.validation_batch_size)
        val_dataset = val_dataset.take(val_steps)  # Limit validation size
        val_dataset = val_dataset.prefetch(self.config.training.prefetch_buffer)
        
        return train_dataset, val_dataset
    
    def train_vae(self, epochs: Optional[int] = None, 
                 save_checkpoints: bool = True) -> Dict:
        """
        Train the VAE model with mixed precision and distributed strategy
        
        Args:
            epochs: Number of epochs (uses config if None)
            save_checkpoints: Whether to save checkpoints
            
        Returns:
            Training history
        """
        # Enable mixed precision training
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        if epochs is None:
            epochs = self.config.training.epochs_per_round
            
        logger.info(f"Training VAE for {epochs} epochs with mixed precision...")

        # Clear memory before preparing data
        import gc
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Prepare distributed datasets
        train_dataset, val_dataset = self.prepare_training_data()

        # Distribute datasets across GPUs
        train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = self.strategy.experimental_distribute_dataset(val_dataset)
        
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
        Paper: 20 rounds, SNR range 10-50 with curriculum learning
        Start with EASIER (full range including strong signals)
        Progress to HARDER (only weak signals)
        """
        if n_rounds is None:
            n_rounds = self.config.training.num_training_rounds
            
        logger.info(f"Starting iterative training for {n_rounds} rounds...")
        
        for round_idx in range(n_rounds):
            logger.info(f"\n{'='*50}")
            logger.info(f"Training round {round_idx + 1}/{n_rounds}")
            logger.info(f"{'='*50}")
            
            # Curriculum learning: gradually reduce the MAXIMUM SNR
            # This forces the model to learn from progressively weaker signals
            
            # Round 1: SNR 10-50 (full range - easy, includes strong signals)
            # Round 10: SNR 10-40 (medium difficulty)  
            # Round 20: SNR 10-30 (hard - only weak signals)
            
            min_snr = 10  # Always start at 10 (paper's minimum)
            
            # Linearly decrease max SNR from 50 to 30 over 20 rounds
            max_snr = 50 - round_idx * (20/n_rounds)  # 50→30
            max_snr = max(30, max_snr)  # Don't go below 30
            
            # Update the config parameters
            self.config.training.snr_base = min_snr
            self.config.training.snr_range = max_snr - min_snr
            
            logger.info(f"SNR range for this round: {min_snr}-{max_snr}")
            logger.info(f"  Easy signals (SNR>40): {max_snr>40}")
            logger.info(f"  Medium signals (SNR 20-40): Always included")  
            logger.info(f"  Hard signals (SNR 10-20): Always included")
            
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
                       n_rounds: int = 20, strategy=None) -> TrainingPipeline:
    """
    Train the complete SETI ML pipeline with distributed strategy
    
    Args:
        config: Configuration object
        background_data: Background observations for data generation
        n_rounds: Number of training rounds (default: 20)
        
    Returns:
        Trained pipeline object
    """
    # Create pipeline
    logger.info("Creating TrainingPipeline...")
    pipeline = TrainingPipeline(config, background_data, strategy=strategy)
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
