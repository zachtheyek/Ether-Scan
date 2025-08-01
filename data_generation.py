"""
Synthetic data generation for SETI ML Pipeline training
"""

import numpy as np
from numba import jit, prange, njit
import setigen as stg
from astropy import units as u
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

@njit(nopython=True)
def compute_signal_line(start_freq: int, drift_rate: float, 
                       time_steps: int, freq_bins: int) -> np.ndarray:
    """
    Compute pixel coordinates for a drifting signal
    
    Args:
        start_freq: Starting frequency bin
        drift_rate: Drift rate in bins per time step
        time_steps: Number of time steps
        freq_bins: Number of frequency bins
        
    Returns:
        Array of (time, freq) coordinates
    """
    coords = np.zeros((time_steps, 2), dtype=np.int32)
    
    for t in range(time_steps):
        freq = int(start_freq + drift_rate * t)
        if 0 <= freq < freq_bins:
            coords[t, 0] = t
            coords[t, 1] = freq
    
    return coords

def inject_signal(data: np.ndarray, snr: float, drift_rate: float,
                 start_freq: Optional[int] = None, 
                 width: float = 50.0) -> Tuple[np.ndarray, float, float]:
    """
    Inject a synthetic narrowband signal into data
    
    Args:
        data: Background data array (time, freq)
        snr: Signal-to-noise ratio
        drift_rate: Drift rate in Hz/s
        start_freq: Starting frequency bin (random if None)
        width: Signal width in Hz
        
    Returns:
        Data with injected signal, drift slope, intercept
    """
    time_steps, freq_bins = data.shape
    
    if start_freq is None:
        # Random start in middle half of band
        start_freq = np.random.randint(freq_bins // 4, 3 * freq_bins // 4)
    
    # Convert drift rate to bins per time step
    freq_resolution = 2.7939677238464355  # Hz per bin
    time_resolution = 18.25361108  # seconds per time step
    drift_bins_per_step = drift_rate * time_resolution / freq_resolution
    
    # Create signal
    signal_power = np.mean(data) * snr
    signal_data = data.copy()
    
    # Get signal path coordinates
    coords = compute_signal_line(start_freq, drift_bins_per_step, time_steps, freq_bins)
    
    # Apply signal with Gaussian profile
    width_bins = int(width / freq_resolution)
    for t, f in coords:
        if f >= 0 and f < freq_bins:
            # Gaussian profile in frequency
            for df in range(-width_bins, width_bins + 1):
                if 0 <= f + df < freq_bins:
                    gaussian_weight = np.exp(-0.5 * (df / (width_bins/3))**2)
                    signal_data[t, f + df] += signal_power * gaussian_weight
    
    # Calculate line parameters for cadence checking
    slope = drift_bins_per_step / freq_bins * time_steps
    intercept = start_freq
    
    return signal_data, slope, intercept

def create_cadence_data(background: np.ndarray, signal_type: str,
                       snr_range: Tuple[float, float] = (10, 50),
                       drift_range: Tuple[float, float] = (-10, 10)) -> np.ndarray:
    """
    Create a full cadence (6 observations) with specified signal type
    
    Args:
        background: Background data array (6, time, freq)
        signal_type: "true" (ETI), "false" (RFI), or "none"
        snr_range: Range of SNR values
        drift_range: Range of drift rates in Hz/s
        
    Returns:
        Cadence data array (6, time, freq)
    """
    cadence = background.copy()
    
    if signal_type == "none":
        return cadence
    
    # Random signal parameters
    snr = np.random.uniform(*snr_range)
    drift_rate = np.random.uniform(*drift_range)
    
    if signal_type == "true":
        # ETI signal: appears only in ON observations (0, 2, 4)
        _, slope, intercept = inject_signal(cadence[0], snr, drift_rate)
        inject_signal(cadence[2], snr, drift_rate, int(intercept))
        inject_signal(cadence[4], snr, drift_rate, int(intercept))
        
    elif signal_type == "false":
        # RFI: appears in all observations
        _, slope, intercept = inject_signal(cadence[0], snr, drift_rate)
        for i in range(1, 6):
            inject_signal(cadence[i], snr, drift_rate, int(intercept))
    
    return cadence

class DataGenerator:
    """Synthetic data generator for training"""
    
    def __init__(self, config, background_plates: np.ndarray):
        """
        Initialize generator
        
        Args:
            config: Configuration object
            background_plates: Array of background observations
        """
        self.config = config
        self.backgrounds = background_plates
        self.n_backgrounds = len(background_plates)
        
    def generate_batch(self, n_samples: int, 
                      signal_type: str = "true") -> np.ndarray:
        """
        Generate batch of synthetic cadences
        
        Args:
            n_samples: Number of samples to generate
            signal_type: Type of signal to inject
            
        Returns:
            Batch of cadences (n_samples, 6, time, freq)
        """
        batch = np.zeros((n_samples, 6, 16, self.config.data.width_bin))
        
        for i in range(n_samples):
            # Random background
            bg_idx = np.random.randint(self.n_backgrounds)
            background = self.backgrounds[bg_idx]
            
            # Generate cadence
            batch[i] = create_cadence_data(
                background, 
                signal_type,
                snr_range=(self.config.training.snr_base, 
                          self.config.training.snr_base + self.config.training.snr_range),
                drift_range=(-self.config.inference.max_drift_rate,
                            self.config.inference.max_drift_rate)
            )
        
        return batch
    
    def generate_training_set(self) -> Dict[str, np.ndarray]:
        """
        Generate complete training dataset
        
        Returns:
            Dictionary with different data types
        """
        n_samples = self.config.training.num_samples_train
        
        logger.info(f"Generating {n_samples} samples per category...")
        
        # Generate different categories
        data = {
            'concatenated': self.generate_batch(n_samples, "true"),
            'true': self.generate_batch(n_samples * 6, "true"),
            'false': self.generate_batch(n_samples * 6, "false"),
            'true_single': self.generate_batch(n_samples * 3, "true"),
            'none': self.generate_batch(n_samples * 3, "none")
        }
        
        # Combine true variations
        data['true_combined'] = np.concatenate([
            data['true_single'],
            self.generate_batch(n_samples * 3, "true")
        ], axis=0)
        
        logger.info(f"Generated training set with {sum(d.shape[0] for d in data.values())} total samples")
        
        return data
    
    def generate_test_set(self) -> Dict[str, np.ndarray]:
        """Generate test dataset"""
        n_samples = self.config.training.num_samples_test
        
        return {
            'true': self.generate_batch(n_samples, "true"),
            'false': self.generate_batch(n_samples, "false"),
            'none': self.generate_batch(n_samples // 2, "none")
        }

def create_mixed_training_batch(generator: DataGenerator, 
                               batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create mixed batch for VAE training
    
    Returns:
        Concatenated data, true cadences, false cadences
    """
    # Generate equal amounts of each type
    n_each = batch_size // 3
    
    true_data = generator.generate_batch(n_each, "true")
    false_data = generator.generate_batch(n_each, "false")
    none_data = generator.generate_batch(n_each, "none")
    
    # Concatenate for main input
    concatenated = np.concatenate([true_data, false_data, none_data], axis=0)
    
    # Additional true/false for clustering loss
    true_clustering = generator.generate_batch(batch_size, "true")
    false_clustering = generator.generate_batch(batch_size, "false")
    
    return concatenated, true_clustering, false_clustering
