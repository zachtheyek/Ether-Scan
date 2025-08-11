"""
Configuration module for SETI ML Pipeline
Contains all hyperparameters and settings - FIXED for proper dimensions
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

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
    width_bin: int = 4096  # Frequency bins per snippet (PAPER SPEC)
    time_bins: int = 16    # Time bins per observation
    downsample_factor: int = 1  # CHANGED: No downsampling to preserve resolution
    num_observations: int = 6  # Per cadence (3 ON, 3 OFF)
    overlap_factor: float = 0.5
    
    # Frequency and time resolution
    freq_resolution: float = 2.7939677238464355  # Hz
    time_resolution: float = 18.25361108  # seconds

    # Data file specifications
    training_files: List[str] = None
    test_files: List[str] = None
    
    # Data subset specifications (for memory management)
    file_subsets: Dict[str, Tuple[Optional[int], Optional[int]]] = None
    
    def __post_init__(self):
        """Initialize default file lists"""
        if self.training_files is None:
            self.training_files = [
                'real_filtered_LARGE_HIP110750.npy',
                'real_filtered_LARGE_HIP13402.npy',
                'real_filtered_LARGE_HIP8497.npy'
            ]
        
        if self.test_files is None:
            self.test_files = [
                'real_filtered_LARGE_test_HIP15638.npy'
            ]
        
        if self.file_subsets is None:
            self.file_subsets = {
                'real_filtered_LARGE_HIP110750.npy': (8000, None),  # Skip first 8000
                'real_filtered_LARGE_HIP13402.npy': (None, 2000),   # REDUCED: Use first 2000 (memory)
                'real_filtered_LARGE_HIP8497.npy': (None, 2000)     # REDUCED: Use first 2000 (memory)
            }
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32   # REDUCED for 4096 frequency bins memory usage
    validation_batch_size: int = 64
    epochs_per_round: int = 50  # REDUCED for initial testing
    num_training_rounds: int = 10  # REDUCED for initial testing
    
    # Data generation parameters  
    num_samples_train: int = 500   # REDUCED from 1000 to avoid OOM with 4096 dims
    num_samples_test: int = 200    # REDUCED from 500
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
    batch_size: int = 1000  # REDUCED for 4096 frequency bins
    max_drift_rate: float = 10.0  # Hz/s
    
class Config:
    """Main configuration class"""
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.rf = RandomForestConfig()
        self.inference = InferenceConfig()
        
        # File paths
        self.data_path = os.environ.get('SETI_DATA_PATH', '/data')
        self.model_path = os.environ.get('SETI_MODEL_PATH', '/models')
        self.output_path = os.environ.get('SETI_OUTPUT_PATH', '/outputs')
        
        # Create paths if they don't exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
    
    def get_training_file_path(self, filename: str) -> str:
        """Get full path for training file"""
        return os.path.join(self.data_path, 'training', filename)
    
    def get_test_file_path(self, filename: str) -> str:
        """Get full path for test file"""
        return os.path.join(self.data_path, 'test', filename)
    
    def get_file_subset(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """Get subset range for file"""
        return self.data.file_subsets.get(filename, (None, None))
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization"""
        return {
            'model': {
                'latent_dim': self.model.latent_dim,
                'dense_layer_size': self.model.dense_layer_size,
                'kernel_size': self.model.kernel_size,
                'alpha': self.model.alpha,
                'beta': self.model.beta,
                'gamma': self.model.gamma,
                'learning_rate': self.model.learning_rate
            },
            'data': {
                'width_bin': self.data.width_bin,
                'time_bins': self.data.time_bins,
                'downsample_factor': self.data.downsample_factor,
                'num_observations': self.data.num_observations
            },
            'training': {
                'batch_size': self.training.batch_size,
                'epochs_per_round': self.training.epochs_per_round,
                'num_training_rounds': self.training.num_training_rounds
            }
        }
