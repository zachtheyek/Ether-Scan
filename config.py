"""
Configuration module for SETI ML Pipeline
Contains all hyperparameters and settings
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
    alpha: float = 1.0   # Reduced clustering loss weight for stability
    beta: float = 0.5    # Reduced KL divergence weight for stability
    gamma: float = 0.0   # Additional loss weight
    learning_rate: float = 0.0001  # More conservative learning rate
    
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
        
        # NOTE: figure out why Peter sliced along these indices specifically
        if self.file_subsets is None:
            self.file_subsets = {
                'real_filtered_LARGE_HIP110750.npy': (8000, None),  # Skip first 8000
                'real_filtered_LARGE_HIP13402.npy': (None, 4000),   # Use first 4000
                'real_filtered_LARGE_HIP8497.npy': (None, 4000)     # Use first 4000
            }
    
@dataclass
class TrainingConfig:
    """Training configuration optimized for 4 GPUs"""
    batch_size: int = 128 # Back to original size after fixing log normalization - Must be divisible by num_gpus (4) 
    validation_batch_size: int = 256 # Back to original size after fixing log normalization - Must be divisible by num_gpus (4)
    epochs_per_round: int = 50
    num_training_rounds: int = 40

    # IMPORTANT: These control memory usage
    samples_per_generator_call: int = 32  # How many samples to generate at once
    prefetch_buffer: int = 2  # How many batches to prefetch
    
    # Data generation parameters  
    num_samples_train: int = 3840 # 30 steps * 128 batch_size
    num_samples_test: int = 7680 # 30 steps * 256 validation_batch_size
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
        
        # Paths - can be overridden by environment variables
        self.data_path = os.environ.get('SETI_DATA_PATH', '/datax/scratch/zachy/data/etherscan')
        self.model_path = os.environ.get('SETI_MODEL_PATH', '/datax/scratch/zachy/models/etherscan')
        self.output_path = os.environ.get('SETI_OUTPUT_PATH', '/datax/scratch/zachy/output/etherscan')

    def get_training_file_path(self, filename: str) -> str:
        """Get full path for a training file"""
        return os.path.join(self.data_path, 'training', filename)
    
    def get_test_file_path(self, filename: str) -> str:
        """Get full path for a test file"""
        return os.path.join(self.data_path, 'testing', filename)
    
    def get_file_subset(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """Get subset slice indices for a file"""
        return self.data.file_subsets.get(filename, (None, None))
        
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
