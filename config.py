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

    # NEW: Memory management parameters
    chunk_size_loading: int = 150  # Cadences to process at once during loading
    max_chunks_per_file: int = 30  # Limit chunks to prevent excessive processing
    
    
    # Training data files (as per paper's training data)
    training_files: List[str] = None
    test_files: List[str] = None
    
    def __post_init__(self):
        """Set default file lists"""
        if self.training_files is None:
            self.training_files = [
                'real_filtered_LARGE_HIP110750.npy',
                'real_filtered_LARGE_HIP13402.npy', 
                'real_filtered_LARGE_HIP8497.npy'
            ]
        if self.test_files is None:
            self.test_files = [
                'real_filtered_LARGE_testHIP83043.npy'
            ]


@dataclass  
class TrainingConfig:
    batch_size: int = 256
    validation_batch_size: int = 128
    num_training_rounds: int = 20
    epochs_per_round: int = 100
    snr_base: int = 10
    snr_range: int = 40
    num_samples_train: int = 5000
    num_samples_test: int = 2000
    num_samples_rf: int = 10000

    # NEW: Memory management parameters
    max_chunk_size: int = 1000  # Maximum samples per chunk during generation
    target_backgrounds: int = 10000  # Number of background cadences to load
    memory_efficient_mode: bool = True  # Enable memory optimizations

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
    
    def get_training_file_path(self, filename: str) -> str:
        """Get full path for training data file"""
        return os.path.join(self.data_path, 'training', filename)
    
    def get_test_file_path(self, filename: str) -> str:
        """Get full path for test data file"""
        return os.path.join(self.data_path, 'testing', filename)
    
    def get_file_subset(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """Get subset parameters for a file (start, end indices)"""
        # Define subsets for specific files to manage memory usage
        subset_map = {
            'real_filtered_LARGE_HIP110750.npy': (12000, 16000),  # As per original code
            'real_filtered_LARGE_HIP13402.npy': (None, 4000),     # First 4000
            'real_filtered_LARGE_HIP8497.npy': (None, 4000),      # First 4000
            'real_filtered_LARGE_testHIP83043.npy': (None, None)  # Full file
        }
        return subset_map.get(filename, (None, None))
    
    def to_dict(self) -> Dict:
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
            'training': {
                'batch_size': self.training.batch_size,
                'validation_batch_size': self.training.validation_batch_size,
                'num_training_rounds': self.training.num_training_rounds,
                'epochs_per_round': self.training.epochs_per_round,
                'snr_base': self.training.snr_base,
                'snr_range': self.training.snr_range,
                'num_samples_train': self.training.num_samples_train,
                'num_samples_test': self.training.num_samples_test
            },
            'data': {
                'width_bin': self.data.width_bin,
                'downsample_factor': self.data.downsample_factor,
                'freq_resolution': self.data.freq_resolution,
                'time_resolution': self.data.time_resolution
            }
        }
