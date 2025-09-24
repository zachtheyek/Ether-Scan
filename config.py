"""
Configuration module for SETI ML Pipeline
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
    beta: float = 1.5    # KL divergence weight
    gamma: float = 0.0   # Not used
    learning_rate: float = 0.001  # Default learning rate
    
@dataclass
class DataConfig:
    """Data processing configuration"""
    width_bin: int = 4096  # Frequency bins per snippet
    time_bins: int = 16    # Time bins per observation
    downsample_factor: int = 8  # Downsampling factor
    num_observations: int = 6  # Per cadence (3 ON, 3 OFF)
    freq_resolution: float = 2.7939677238464355  # Hz
    time_resolution: float = 18.25361108  # seconds

    # Memory management parameters
    chunk_size_loading: int = 150  # Cadences to process at once during background loading
    max_chunks_per_file: int = 30  # Max backgrounds per file = max_chunks_per_file * chunk_size_loading
    
    # Data files
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
    num_training_rounds: int = 20
    epochs_per_round: int = 100

    # BUG: 
    # Gradient accumulation bug causes num_replicas small updates instead of 1 large update
    # Fix later; for now, set train_physical_batch_size = train_logical_batch_size as temp patch
    train_physical_batch_size: int = 1024  # Micro batch size for memory efficiency
    train_logical_batch_size: int = 1024  # Actual batch size for convergence 
    validation_batch_size: int = 4096

    target_backgrounds: int = 10000  # Number of background cadences to load
    max_chunk_size: int = 1000  # Maximum samples per chunk during generation
    num_samples_train: int = 5000
    num_samples_test: int = 2000
    num_samples_rf: int = 10000
    train_val_split: float = 0.8

    # Curriculum learning parameters
    snr_base: int = 10 
    initial_snr_range: int = 40
    final_snr_range: int = 20
    curriculum_schedule: str = "exponential"  # "linear", "exponential", "step"
    exponential_decay_rate: int = -3  # How quickly training should progress from easy to hard (must be <0) (more negative = less easy rounds & more hard rounds)
    step_easy_rounds: int = 5  # Number of rounds with easy signals
    step_hard_rounds: int = 15  # Number of rounds with challenging signals

    # Fault tolerance parameters
    max_retries: int = 5 
    retry_delay: int = 30 # seconds 

# NOTE: come back to this later
@dataclass
class RandomForestConfig:
    """Random Forest configuration"""
    n_estimators: int = 1000
    bootstrap: bool = True
    max_features: str = 'sqrt'
    n_jobs: int = -1

# NOTE: come back to this later
@dataclass
class InferenceConfig:
    """Inference configuration"""
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
            'real_filtered_LARGE_HIP110750.npy': (None, 5000),  # First 5000
            'real_filtered_LARGE_HIP13402.npy': (8000, 10000),  # Middle 2000
            'real_filtered_LARGE_HIP8497.npy': (11000, None),  # Last 3567
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
            'data': {
                'width_bin': self.data.width_bin,
                'time_bins': self.data.time_bins,
                'downsample_factor': self.data.downsample_factor,
                'num_observations': self.data.num_observations,
                'freq_resolution': self.data.freq_resolution,
                'time_resolution': self.data.time_resolution,
                'chunk_size_loading': self.data.chunk_size_loading,
                'max_chunks_per_file': self.data.max_chunks_per_file,
                'training_files': self.data.training_files,
                'test_files': self.data.test_files
            },
            'training': {
                'num_training_rounds': self.training.num_training_rounds,
                'epochs_per_round': self.training.epochs_per_round,
                'train_physical_batch_size': self.training.train_physical_batch_size,
                'train_logical_batch_size': self.training.train_logical_batch_size,
                'validation_batch_size': self.training.validation_batch_size,
                'target_backgrounds': self.training.target_backgrounds,
                'max_chunk_size': self.training.max_chunk_size,
                'num_samples_train': self.training.num_samples_train,
                'num_samples_test': self.training.num_samples_test,
                'num_samples_rf': self.training.num_samples_rf,
                'train_val_split': self.training.train_val_split,
                'snr_base': self.training.snr_base,
                'initial_snr_range': self.training.initial_snr_range,
                'final_snr_range': self.training.final_snr_range,
                'curriculum_schedule': self.training.curriculum_schedule,
                'exponential_decay_rate': self.training.exponential_decay_rate,
                'step_easy_rounds': self.training.step_easy_rounds,
                'step_hard_rounds': self.training.step_hard_rounds,
                'max_retries': self.training.max_retries,
                'retry_delay': self.training.retry_delay
            },
            'rf': {
                'n_estimators': self.rf.n_estimators,
                'bootstrap': self.rf.bootstrap,
                'max_features': self.rf.max_features,
                'n_jobs': self.rf.n_jobs
            },
            'inference': {
                'classification_threshold': self.inference.classification_threshold,
                'batch_size': self.inference.batch_size,
                'max_drift_rate': self.inference.max_drift_rate
            },
            'paths': {
                'data_path': self.data_path,
                'model_path': self.model_path,
                'output_path': self.output_path
            }
        }
