"""
Synthetic data generation for SETI ML Pipeline
Fixed to match author's approach using setigen
"""

import numpy as np
from numba import jit, prange, njit
import setigen as stg
from astropy import units as u
from typing import Tuple, Optional, Dict, List
import logging
from random import random
import math

logger = logging.getLogger(__name__)

def new_cadence(data: np.ndarray, snr: float, width_bin: int = 512) -> Tuple[np.ndarray, float, float]:
    """
    Create a cadence with injected signal following author's approach
    This matches the author's new_cadence function
    
    Args:
        data: Background data array (96, width_bin) 
        snr: Signal-to-noise ratio
        width_bin: Number of frequency bins (512 after downsampling)
        
    Returns:
        Data with injected signal, slope, intercept
    """
    CONST = 3
    start = int(random() * (width_bin - 1)) + 1
    
    if (-1)**(int(random()*3+1)) > 0:
        true_slope = (96/start)
        slope = (true_slope) * (18.25361108/2.7939677238464355) + random()*CONST
    else:
        true_slope = (96/(start-width_bin))
        slope = (true_slope) * (18.25361108/2.7939677238464355) - random()*CONST
    
    drift = -1*(1/slope)
    
    # Width calculation as per author
    width = random()*50 + abs(drift)*18./1
    
    b = 96 - true_slope*(start)
    
    # Use setigen Frame as the author does
    frame = stg.Frame.from_data(
        df=2.7939677238464355*u.Hz,
        dt=18.25361108*u.s,
        fch1=0*u.MHz,
        data=data,
        ascending=True
    )
    
    signal = frame.add_signal(
        stg.constant_path(
            f_start=frame.get_frequency(index=start),
            drift_rate=drift*u.Hz/u.s
        ),
        stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
        stg.gaussian_f_profile(width=width*u.Hz),
        stg.constant_bp_profile(level=1)
    )
    
    return frame.data, true_slope, b

def intersection(m1, m2, b1, b2):
    """Check if two drifting signals intersect in valid regions"""
    solution = (b2-b1)/(m1-m2)
    y = m1*solution + b1
    
    # Check if intersection is in OFF regions (should return False)
    if y >= 80 and y <= 96:
        return False
    elif y >= 64 and y <= 80:
        return False
    elif y >= 32 and y <= 48:
        return False
    elif y >= 0 and y <= 16:
        return False
    else:
        return True

def create_true(plate: np.ndarray, snr_base: float = 300, snr_range: float = 10, 
                factor: float = 1, index: int = 1, width_bin: int = 512) -> np.ndarray:
    """
    Create true ETI signal following author's exact approach
    Signal appears in ON observations (0, 2, 4) with consistent drift
    
    Args:
        plate: Background data (n_backgrounds, 6, 16, width_bin)
        snr_base: Base SNR
        snr_range: SNR range
        factor: Multiplication factor for second signal
        index: Not used (for compatibility)
        width_bin: Frequency bins
        
    Returns:
        Cadence with true ETI signal (6, 16, width_bin)
    """
    # Random background selection
    index = int(plate.shape[0] * random())
    total = np.zeros((6, plate.shape[2], plate.shape[3]))
    base = plate[index, :, :, :]
    
    # Combine into single 96x512 array for signal injection
    data = np.zeros((96, width_bin))
    for el in range(6):
        data[16*el:(el+1)*16, :] = base[el, :, :]
    
    # Keep trying until we get non-intersecting signals
    while True:
        snr = random() * snr_range + snr_base
        cadence, m1, b1 = new_cadence(data, snr, width_bin)
        injection_plate, m2, b2 = new_cadence(cadence, snr*factor, width_bin)
        
        if m1 != m2:
            if intersection(m1, m2, b1, b2):
                break
    
    # Split back into 6 observations
    total[0, :, :] = injection_plate[0:16, :]
    total[1, :, :] = cadence[16:32, :]
    total[2, :, :] = injection_plate[32:48, :]
    total[3, :, :] = cadence[48:64, :]
    total[4, :, :] = injection_plate[64:80, :]
    total[5, :, :] = cadence[80:96, :]
    
    return total

def create_true_single_shot(plate: np.ndarray, snr_base: float = 10, snr_range: float = 5,
                            factor: float = 1, index: int = 1, width_bin: int = 512) -> np.ndarray:
    """
    Create single-shot true signal (appears in ON observations only)
    """
    index = int(plate.shape[0] * random())
    total = np.zeros((6, plate.shape[2], plate.shape[3]))
    base = plate[index, :, :, :]
    
    data = np.zeros((96, width_bin))
    for el in range(6):
        data[16*el:(el+1)*16, :] = base[el, :, :]
    
    snr = random() * snr_range + snr_base
    injection_plate, m2, b2 = new_cadence(data, snr, width_bin)
    
    total[0, :, :] = injection_plate[0:16, :]
    total[1, :, :] = data[16:32, :]  # OFF - no signal
    total[2, :, :] = injection_plate[32:48, :]
    total[3, :, :] = data[48:64, :]  # OFF - no signal
    total[4, :, :] = injection_plate[64:80, :]
    total[5, :, :] = data[80:96, :]  # OFF - no signal
    
    return total

def create_false(plate: np.ndarray, snr_base: float = 300, snr_range: float = 10,
                factor: float = 1, index: int = 1, width_bin: int = 512) -> np.ndarray:
    """
    Create false signal (RFI) - appears in all observations or none
    """
    choice = random()
    
    if choice > 0.5:
        # Inject RFI in all observations
        index = int(plate.shape[0] * random())
        total = np.zeros((6, plate.shape[2], plate.shape[3]))
        base = plate[index, :, :, :]
        
        data = np.zeros((96, width_bin))
        for el in range(6):
            data[16*el:(el+1)*16, :] = base[el, :, :]
        
        snr = random() * snr_range + snr_base
        cadence, m1, b1 = new_cadence(data, snr, width_bin)
        
        # RFI appears in all observations
        for i in range(6):
            total[i, :, :] = cadence[i*16:(i+1)*16, :]
    else:
        # Just return background
        index = int(plate.shape[0] * random())
        total = plate[index, :, :, :]
    
    return total

@jit(nopython=True, parallel=True)
def create_full_cadence(function, samples: int, plate: np.ndarray, 
                        snr_base: float = 300, snr_range: float = 10,
                        factor: float = 1, width_bin: int = 512) -> np.ndarray:
    """
    Create multiple cadences in parallel
    """
    data = np.zeros((samples, 6, 16, width_bin))
    
    for i in prange(samples):
        data[i, :, :, :] = function(plate, snr_base=snr_base, snr_range=snr_range,
                                   factor=factor, width_bin=width_bin)
    
    return data

def create_mixed_training_batch(data_generator, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a mixed training batch for VAE training
    Wrapper function to match training pipeline expectations
    
    Args:
        data_generator: DataGenerator instance
        batch_size: Size of batch to generate
        
    Returns:
        Tuple of (concatenated, true, false) data arrays
    """
    batch_data = data_generator.generate_training_batch(batch_size)
    
    return (
        batch_data['concatenated'],
        batch_data['true'], 
        batch_data['false']
    )

class DataGenerator:
    """Synthetic data generator matching author's approach"""
    
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
        
        # Pre-compute width_bin from config
        self.width_bin = config.data.width_bin 
        
        logger.info(f"DataGenerator initialized with {self.n_backgrounds} background plates")
        logger.info(f"Background shape: {background_plates.shape}")
        logger.info(f"Using full width_bin: {self.width_bin} (will downsample later)")
        
    def generate_training_batch(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate training batch following author's exact approach
        
        The author creates:
        - 1/4 false (no signal)
        - 1/4 true single shot
        - 1/4 true double shot (with RFI)
        - 1/4 false (with RFI)
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            Dictionary with concatenated, true, and false data
        """
        quarter = n_samples // 4
        
        # Generate each type
        false_no_signal = create_full_cadence(
            create_false, quarter, self.backgrounds,
            snr_base=self.config.training.snr_base,
            snr_range=self.config.training.snr_range,
            width_bin=self.width_bin
        )
        
        true_single = create_full_cadence(
            create_true_single_shot, quarter, self.backgrounds,
            snr_base=self.config.training.snr_base,
            snr_range=self.config.training.snr_range,
            width_bin=self.width_bin
        )
        
        true_double = create_full_cadence(
            create_true, quarter, self.backgrounds,
            snr_base=self.config.training.snr_base,
            snr_range=self.config.training.snr_range,
            factor=1,  # Same intensity for both signals
            width_bin=self.width_bin
        )
        
        false_with_rfi = create_full_cadence(
            create_false, quarter, self.backgrounds,
            snr_base=self.config.training.snr_base,
            snr_range=self.config.training.snr_range,
            width_bin=self.width_bin
        )
        
        # Concatenate for main training data
        concatenated = np.concatenate([
            false_no_signal,
            true_single,
            true_double,
            false_with_rfi
        ], axis=0)
        
        # Generate separate true/false for clustering loss
        true_clustering = create_full_cadence(
            create_true, n_samples, self.backgrounds,
            snr_base=self.config.training.snr_base,
            snr_range=self.config.training.snr_range,
            width_bin=self.width_bin
        )
        
        false_clustering = create_full_cadence(
            create_false, n_samples, self.backgrounds,
            snr_base=self.config.training.snr_base,
            snr_range=self.config.training.snr_range,
            width_bin=self.width_bin
        )
        
        return {
            'concatenated': concatenated.astype(np.float32),
            'true': true_clustering.astype(np.float32),
            'false': false_clustering.astype(np.float32)
        }
    
    def generate_test_batch(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate test batch with balanced classes"""
        n_each = n_samples // 3
        
        false_data = create_full_cadence(
            create_false, n_each, self.backgrounds,
            snr_base=self.config.training.snr_base,
            snr_range=self.config.training.snr_range,
            width_bin=self.width_bin
        )
        
        true_single = create_full_cadence(
            create_true_single_shot, n_each, self.backgrounds,
            snr_base=self.config.training.snr_base,
            snr_range=self.config.training.snr_range,
            width_bin=self.width_bin
        )
        
        true_double = create_full_cadence(
            create_true, n_each, self.backgrounds,
            snr_base=self.config.training.snr_base,
            snr_range=self.config.training.snr_range,
            width_bin=self.width_bin
        )
        
        return {
            'false': false_data.astype(np.float32),
            'true_single': true_single.astype(np.float32),
            'true_double': true_double.astype(np.float32)
        }
