"""
Configuration module for SETI ML Pipeline
Updated to match author's exact hyperparameters
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

@dataclass
class ModelConfig:
    """VAE model configuration - author's exact settings"""
    latent_dim: int = 8
    dense_layer_size: int = 512
    kernel_size: Tuple[int, int] = (3, 3)
    alpha: float = 10.0  # Clustering loss weight (author's value)
    beta: float = 1.5    # KL divergence weight (author's value)
    gamma: float = 0.0   # Not used in paper
    learning_rate: float = 0.001  # Author's learning rate
    
@dataclass
class DataConfig:
    """Data processing configuration"""
    width_bin: int = 4096  # Frequency bins per snippet
    time_bins: int = 16    # Time bins per observation
    downsample_factor: int = 8  # Downsampling factor
    num_observations: int = 6  # Per cadence (3 ON, 3 OFF)
    
    # Frequency and time resolution from paper
    freq_resolution: float = 2.7939677238464355  # Hz
    time_resolution: float = 18.25361108  # seconds

@dataclass
class TrainingConfig:
    """Training configuration matching author's approach"""
    # Author uses varying batch sizes 1000-2000
    batch_size: int = 1000
    validation_batch_size: int = 500
    
    # Training rounds from paper
    num_training_rounds: int = 20
    
    # SNR parameters - consistent across all rounds
    snr_base: int = 10
    snr_range: int = 40  # So SNR is 10-50
    
    # Sample counts for data generation
    num_samples_train: int = 5000  # Per type (true/false/single)
    num_samples_rf: int = 12000   # For Random Forest training

@dataclass
class RandomForestConfig:
    """Random Forest configuration from paper"""
    n_estimators: int = 1000
    bootstrap: bool = True
    max_features: str = 'sqrt'
    n_jobs: int = -1

@dataclass
class InferenceConfig:
    """Inference configuration"""
    classification_threshold: float = 0.5
    batch_size: int = 5000  # Author's inference batch size
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
        self.data_path = os.environ.get('SETI_DATA_PATH', '/datax/scratch/zachy/data/etherscan')
        self.model_path = os.environ.get('SETI_MODEL_PATH', '/datax/scratch/zachy/models/etherscan')
        self.output_path = os.environ.get('SETI_OUTPUT_PATH', '/datax/scratch/zachy/output/etherscan')

