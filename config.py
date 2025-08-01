"""
Configuration module for SETI ML Pipeline
Contains all hyperparameters and settings
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ModelConfig:
    """VAE model configuration"""
    latent_dim: int = 8
    dense_layer_size: int = 512
    kernel_size: Tuple[int, int] = (3, 3)
    alpha: float = 10.0  # Clustering loss weight
    beta: float = 1.5   # KL divergence weight
    gamma: float = 0.0   # Additional loss weight
    learning_rate: float = 0.001
    
@dataclass
class DataConfig:
    """Data processing configuration"""
    width_bin: int = 4096  # Frequency bins per snippet
    time_bins: int = 16    # Time bins per observation
    downsample_factor: int = 8
    num_observations: int = 6  # Per cadence (3 ON, 3 OFF)
    overlap_factor: float = 0.5
    
    # Frequency and time resolution
    freq_resolution: float = 2.7939677238464355  # Hz
    time_resolution: float = 18.25361108  # seconds
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 1000
    validation_batch_size: int = 6000
    epochs_per_round: int = 100
    num_training_rounds: int = 20
    
    # Data generation parameters
    num_samples_train: int = 6000
    num_samples_test: int = 1000
    snr_base: int = 10
    snr_range: int = 40
    
@dataclass
class RandomForestConfig:
    """Random Forest configuration"""
    n_estimators: int = 1000
    bootstrap: bool = True
    max_features: str = 'sqrt'
    n_jobs: int = -1
    
@dataclass
class InferenceConfig:
    """Inference/execution configuration"""
    classification_threshold: float = 0.5
    batch_size: int = 5000
    max_drift_rate: float = 10.0  # Hz/s
    
class Config:
    """Main configuration class"""
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.rf = RandomForestConfig()
        self.inference = InferenceConfig()
        
        # Paths
        self.data_path = os.environ.get('SETI_DATA_PATH', '/data/seti')
        self.model_path = os.environ.get('SETI_MODEL_PATH', '/models/seti')
        self.output_path = os.environ.get('SETI_OUTPUT_PATH', '/output/seti')
        
    def to_dict(self):
        """Convert config to dictionary for serialization"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'rf': self.rf.__dict__,
            'inference': self.inference.__dict__,
            'paths': {
                'data': self.data_path,
                'model': self.model_path,
                'output': self.output_path
            }
        }
