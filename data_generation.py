"""
Synthetic data generation for SETI ML Pipeline training
Fixed drift rate bias and signal injection logic
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
    Paper: Signals with SNR 10-50, drift rate ±10 Hz/s, width 5-55 Hz
    
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
        # Random start in middle half of band to avoid edge effects
        start_freq = np.random.randint(freq_bins // 4, 3 * freq_bins // 4)
    
    # Convert drift rate to bins per time step
    # Paper: 2.79 Hz native resolution, 8x downsample = 22.32 Hz per bin
    freq_resolution = 22.32  # Hz per downsampled bin
    time_resolution = 18.25361108  # seconds per time step
    drift_bins_per_step = drift_rate * time_resolution / freq_resolution
    
    # Create signal
    noise_std = np.std(data)
    signal_power = noise_std * snr
    signal_data = data.copy()
    
    # Get signal path coordinates
    coords = compute_signal_line(start_freq, drift_bins_per_step, time_steps, freq_bins)
    
    # Apply signal with Gaussian profile (narrowband)
    width_bins = max(1, int(width / freq_resolution))
    
    for t, f in coords:
        if f >= 0 and f < freq_bins:
            # Gaussian profile in frequency
            for df in range(-width_bins, width_bins + 1):
                if 0 <= f + df < freq_bins:
                    gaussian_weight = np.exp(-0.5 * (df / (width_bins/3))**2)
                    signal_data[t, f + df] += signal_power * gaussian_weight
    
    # Calculate line parameters for verification
    slope = drift_bins_per_step
    intercept = start_freq
    
    return signal_data, slope, intercept

def create_cadence_data(background: np.ndarray, signal_type: str,
                       snr_range: Tuple[float, float] = (10, 50),
                       drift_range: Tuple[float, float] = (-10, 10)) -> np.ndarray:
    """
    Create a full cadence (6 observations) with specified signal type
    Paper: ETI appears in ON (0,2,4), RFI appears in all
    
    Args:
        background: Background data array (6, time, freq)
        signal_type: "true" (ETI), "false" (RFI), or "none"
        snr_range: Range of SNR values (10-50 per paper)
        drift_range: Range of drift rates in Hz/s (±10 per paper)
        
    Returns:
        Cadence data array (6, time, freq)
    """
    cadence = background.copy()
    
    if signal_type == "none":
        return cadence
    
    # Random signal parameters
    snr = np.random.uniform(*snr_range)
    # FIXED: Use uniform distribution for drift rate (no bias)
    drift_rate = np.random.uniform(*drift_range)
    
    # Width calculation per paper
    width = np.random.uniform(5, 55)  # Hz
    
    if signal_type == "true":
        # ETI signal: appears only in ON observations (0, 2, 4)
        _, slope, intercept = inject_signal(cadence[0], snr, drift_rate, width=width)
        inject_signal(cadence[2], snr, drift_rate, int(intercept), width=width)
        inject_signal(cadence[4], snr, drift_rate, int(intercept), width=width)
        
    elif signal_type == "false":
        # RFI: appears in all 6 observations
        _, slope, intercept = inject_signal(cadence[0], snr, drift_rate, width=width)
        for i in range(1, 6):
            inject_signal(cadence[i], snr, drift_rate, int(intercept), width=width)
    
    return cadence

class DataGenerator:
    """Synthetic data generator for training"""
    
    def __init__(self, config, background_plates: np.ndarray):
        """
        Initialize generator
        
        Args:
            config: Configuration object
            background_plates: Array of background observations
                              Shape: (n_backgrounds, 6, 16, 512) after preprocessing
        """
        self.config = config
        self.backgrounds = background_plates
        self.n_backgrounds = len(background_plates)
        logger.info(f"DataGenerator initialized with {self.n_backgrounds} background plates")
        logger.info(f"Background shape: {background_plates.shape if len(background_plates) > 0 else 'empty'}")
        
    def generate_batch(self, n_samples: int, 
                      signal_type: str = "true") -> np.ndarray:
        """
        Generate batch of synthetic cadences
        
        Args:
            n_samples: Number of samples to generate
            signal_type: Type of signal to inject ("true", "false", "none")
            
        Returns:
            Batch of cadences (n_samples, 6, 16, 512)
        """
        # Get dimensions from config
        time_bins = 16
        freq_bins = self.config.data.width_bin // self.config.data.downsample_factor  # 512
        
        batch = np.zeros((n_samples, 6, time_bins, freq_bins), dtype=np.float32)
        
        for i in range(n_samples):
            # Random background selection
            bg_idx = np.random.randint(self.n_backgrounds)
            background = self.backgrounds[bg_idx]
            
            # Generate cadence with signal injection
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
        Paper: 120,000 samples each for concatenated, true, false
        
        Returns:
            Dictionary with different data types
        """
        # Per paper: 120,000 samples for main training
        n_samples = 120000
        
        logger.info(f"Generating training set with {n_samples} samples per category...")
        
        # Generate different categories as per paper Figure 1
        data = {
            # Main concatenated training data (1/4 none, 1/4 true single, 1/4 true+RFI, 1/4 RFI)
            'concatenated': np.concatenate([
                self.generate_batch(n_samples // 4, "none"),
                self.generate_batch(n_samples // 4, "true"),
                self.generate_batch(n_samples // 4, "true"),  # Will add RFI in post
                self.generate_batch(n_samples // 4, "false")
            ], axis=0),
            
            # Separate true/false for clustering loss
            'true': self.generate_batch(n_samples, "true"),
            'false': self.generate_batch(n_samples, "false")
        }
        
        # Add RFI to third quarter of concatenated
        for i in range(n_samples // 2, 3 * n_samples // 4):
            # Add RFI signal to existing true signal
            data['concatenated'][i] = create_cadence_data(
                data['concatenated'][i], "false",
                snr_range=(10, 30),  # Lower SNR for RFI
                drift_range=(-5, 5)   # Smaller drift for RFI
            )
        
        total_samples = sum(d.shape[0] for d in data.values())
        logger.info(f"Generated training set with {total_samples} total samples")
        
        return data
    
    def generate_test_set(self) -> Dict[str, np.ndarray]:
        """Generate test dataset (24,000 samples as per paper)"""
        n_samples = 24000 // 3  # Divide by 3 for balanced classes
        
        return {
            'true': self.generate_batch(n_samples, "true"),
            'false': self.generate_batch(n_samples, "false"),
            'none': self.generate_batch(n_samples, "none")
        }

def create_mixed_training_batch(generator: DataGenerator, 
                               batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create mixed batch for VAE training
    
    Returns:
        Concatenated data, true cadences, false cadences
    """
    # Generate equal amounts of each type for balanced training
    n_each = batch_size // 4
    
    none_data = generator.generate_batch(n_each, "none")
    true_data = generator.generate_batch(n_each, "true")
    false_data = generator.generate_batch(n_each, "false")
    mixed_data = generator.generate_batch(n_each, "true")  # Will add RFI
    
    # Add RFI to mixed data
    for i in range(n_each):
        mixed_data[i] = create_cadence_data(
            mixed_data[i], "false",
            snr_range=(10, 30),
            drift_range=(-5, 5)
        )
    
    # Concatenate for main input
    concatenated = np.concatenate([none_data, true_data, mixed_data, false_data], axis=0)
    
    # Additional true/false for clustering loss
    true_clustering = generator.generate_batch(batch_size, "true")
    false_clustering = generator.generate_batch(batch_size, "false")
    
    return concatenated, true_clustering, false_clustering
