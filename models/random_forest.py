# NOTE: come back to this later
"""
Random Forest classifier for SETI ML Pipeline
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import joblib
from typing import Tuple, Optional
import logging
from numba import prange

logger = logging.getLogger(__name__)

def prepare_latent_features(latent_vectors: np.ndarray) -> np.ndarray:
    """
    Prepare latent vectors for Random Forest input
    Concatenates the 6 observation latent vectors per cadence
    
    Args:
        latent_vectors: Array of shape (n_samples*6, latent_dim)
        
    Returns:
        Array of shape (n_samples, 6*latent_dim)
    """
    n_total = latent_vectors.shape[0]
    n_samples = n_total // 6
    latent_dim = latent_vectors.shape[1]
    
    features = np.zeros((n_samples, 6 * latent_dim))
    
    for i in range(n_samples):
        # Concatenate all 6 observation latent vectors
        features[i, :] = latent_vectors[i*6:(i+1)*6, :].ravel()
    
    return features

class RandomForestModel:
    """Random Forest classifier for SETI signal detection"""
    
    def __init__(self, config):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.rf.n_estimators,
            bootstrap=config.rf.bootstrap,
            max_features=config.rf.max_features,
            n_jobs=config.rf.n_jobs,
            random_state=42
        )
        self.is_trained = False
        
    def prepare_training_data(self, true_latents: np.ndarray, 
                            false_latents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for Random Forest
        
        Args:
            true_latents: Latent vectors from true (ETI) signals
            false_latents: Latent vectors from false (RFI/noise) signals
            
        Returns:
            Features array and labels array
        """
        # Prepare features
        true_features = prepare_latent_features(true_latents)
        false_features = prepare_latent_features(false_latents)
        
        # Combine features
        features = np.concatenate([true_features, false_features], axis=0)
        
        # Create labels
        labels = np.concatenate([
            np.ones(true_features.shape[0]),
            np.zeros(false_features.shape[0])
        ])
        
        # Shuffle data
        features, labels = shuffle(features, labels, random_state=42)
        
        logger.info(f"Prepared {features.shape[0]} training samples")
        
        return features, labels
    
    def train(self, true_latents: np.ndarray, false_latents: np.ndarray):
        """
        Train the Random Forest model
        
        Args:
            true_latents: Latent vectors from true signals
            false_latents: Latent vectors from false signals
        """
        features, labels = self.prepare_training_data(true_latents, false_latents)
        
        logger.info("Training Random Forest classifier...")
        self.model.fit(features, labels)
        self.is_trained = True
        
        # Log feature importances
        importances = self.model.feature_importances_
        logger.info(f"Feature importance stats - Mean: {np.mean(importances):.4f}, "
                   f"Std: {np.std(importances):.4f}")
    
    def predict_proba(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for cadences
        
        Args:
            latent_vectors: Latent vectors from encoder
            
        Returns:
            Probability array of shape (n_samples, 2)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        features = prepare_latent_features(latent_vectors)
        return self.model.predict_proba(features)
    
    def predict(self, latent_vectors: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict classes for cadences
        
        Args:
            latent_vectors: Latent vectors from encoder
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        probas = self.predict_proba(latent_vectors)
        return (probas[:, 1] > threshold).astype(int)
    
    def evaluate_cadence_pattern(self, probas: np.ndarray, 
                                threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate cadence patterns for strong signal detection
        
        Args:
            probas: Probability array
            threshold: Classification threshold
            
        Returns:
            Binary decisions and confidence scores
        """
        # Check if probability exceeds threshold
        decisions = probas[:, 1] > threshold
        
        # Confidence score is the probability of the predicted class
        confidences = np.where(decisions, probas[:, 1], probas[:, 0])
        
        return decisions, confidences
    
    def save(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Saved Random Forest model to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Loaded Random Forest model from {filepath}")

def train_random_forest(vae_encoder, training_data, config):
    """
    Train Random Forest classifier using VAE encoder
    
    Args:
        vae_encoder: Trained VAE encoder model
        training_data: Dict with 'true' and 'false' data arrays
        config: Configuration object
        
    Returns:
        Trained RandomForestModel
    """
    rf_model = RandomForestModel(config)
    
    # Extract latent representations
    logger.info("Extracting latent representations...")
    
    true_latents = vae_encoder.predict(
        training_data['true'], 
        batch_size=config.training.train_logical_batch_size
    )[2]  # Get z vectors
    
    false_latents = vae_encoder.predict(
        training_data['false'],
        batch_size=config.training.train_logical_batch_size
    )[2]
    
    # Train Random Forest
    rf_model.train(true_latents, false_latents)
    
    return rf_model
