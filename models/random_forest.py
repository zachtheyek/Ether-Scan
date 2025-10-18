"""
Random Forest classifier for SETI ML Pipeline
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import joblib
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def prepare_latent_features(latent_vectors: np.ndarray, num_observations: int = 6) -> np.ndarray:
    """
    Prepare latent vectors for Random Forest input
    Recombines the latent vectors into their original 6-observation cadence pattern
    """
    # Expected shape: (num_cadences * num_observations, latent_dim)
    num_latents = latent_vectors.shape[0]

    if num_latents % num_observations != 0:
        raise ValueError(f"Received {num_latents} latent vectors. Not divisible by num_observations ({num_observations})")

    num_cadences = num_latents // num_observations
    latent_dim = latent_vectors.shape[1]
    
    # Target shape: (num_cadences, num_observations * latent_dim)
    # Where each element in the latent vector is treated as a feature by the Random Forest
    # We flatten the observations so all 6 latents in a cadence are grouped together 
    features = np.zeros((num_cadences, num_observations * latent_dim))
    
    for i in range(num_cadences):
        # Flatten & concatenate the latent vectors according to the number of observations 
        features[i, :] = latent_vectors[i*num_observations:(i+1)*num_observations, :].ravel()
    
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
            random_state=config.rf.seed
        )
        self.is_trained = False
        
    def prepare_training_data(self, true_latents: np.ndarray, 
                              false_latents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for Random Forest
        Combines true/false features, generates labels, and shuffles data
        """
        # Prepare features
        true_features = prepare_latent_features(true_latents, self.config.data.num_observations)
        false_features = prepare_latent_features(false_latents, self.config.data.num_observations)
        
        # Combine features
        features = np.concatenate([true_features, false_features], axis=0)
        
        # Create labels
        labels = np.concatenate([
            np.ones(true_features.shape[0]),
            np.zeros(false_features.shape[0])
        ])
        
        # Shuffle data
        features, labels = shuffle(features, labels, random_state=self.config.rf.seed)
        
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
        
    # NOTE: pick back up from commented sections. focus on creating visualizations for training performance (evaluation.py?)
    #     # Log feature importances
    #     importances = self.model.feature_importances_
    #     logger.info(f"Feature importance stats - Mean: {np.mean(importances):.4f}, "
    #                f"Std: {np.std(importances):.4f}")
    #
    # def predict_proba(self, latent_vectors: np.ndarray) -> np.ndarray:
    #     """
    #     Predict probabilities for cadences
    #
    #     Args:
    #         latent_vectors: Latent vectors from encoder
    #
    #     Returns:
    #         Probability array of shape (n_samples, 2)
    #     """
    #     if not self.is_trained:
    #         raise RuntimeError("Model must be trained before prediction")
    #
    #     features = prepare_latent_features(latent_vectors)
    #     return self.model.predict_proba(features)
    #
    # def predict(self, latent_vectors: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    #     """
    #     Predict classes for cadences
    #
    #     Args:
    #         latent_vectors: Latent vectors from encoder
    #         threshold: Classification threshold
    #
    #     Returns:
    #         Binary predictions
    #     """
    #     probas = self.predict_proba(latent_vectors)
    #     return (probas[:, 1] > threshold).astype(int)
    #
    # def evaluate_cadence_pattern(self, probas: np.ndarray, 
    #                             threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Evaluate cadence patterns for strong signal detection
    #
    #     Args:
    #         probas: Probability array
    #         threshold: Classification threshold
    #
    #     Returns:
    #         Binary decisions and confidence scores
    #     """
    #     # Check if probability exceeds threshold
    #     decisions = probas[:, 1] > threshold
    #
    #     # Confidence score is the probability of the predicted class
    #     confidences = np.where(decisions, probas[:, 1], probas[:, 0])
    #
    #     return decisions, confidences
    
    def save(self, filepath: str):
        """Save RF model weights"""
        if not self.is_trained:
            logger.warning("Saving untrained model")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Saved Random Forest model to {filepath}")
    
    def load(self, filepath: str):
        """Load RF model weights"""
        if self.is_trained:
            logger.warning("Overriding trained model")

        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Loaded Random Forest model from {filepath}")
