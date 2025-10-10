"""
Data preprocessing module for SETI ML Pipeline
"""

import numpy as np
from numba import jit, prange
from skimage.transform import downscale_local_mean
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

def pre_proc(data: np.ndarray) -> np.ndarray:
    """
    Apply log normalization to data
    """
    # Add small epsilon to avoid log(0)
    data = data + 1e-10
    
    # Normalization sequence
    data = np.log(data)
    data = data - data.min()
    if data.max() > 0:
        data = data / data.max()
    return data

# NOTE: come back to this later (start here)
# NOTE: preprocess_cadence() is used in inference.py, everything else unused? 
# NOTE: maybe repurpose preprocess.py to do bandpass removal & energy detection? 

@jit(nopython=True, parallel=True)
def shaping_data_dynamic(data: np.ndarray, width_bin: int = 4096) -> np.ndarray:
    """
    Reshape raw observation data into snippets
    
    Args:
        data: Raw observation data (time_bins, polarization, frequency_channels)
        width_bin: Number of frequency bins per snippet (4096 as per paper)
        
    Returns:
        Reshaped data (num_snippets, time_bins, freq_bins, 1)
    """
    # Handle both single and dual polarization
    if len(data.shape) == 3:
        time_bins, n_pol, total_freq = data.shape
        # Use first polarization only
        data_to_use = data[:, 0, :]
    else:
        # Single polarization case
        time_bins, total_freq = data.shape
        data_to_use = data
    
    samples = total_freq // width_bin
    new_data = np.zeros((samples, time_bins, width_bin, 1))
    
    for i in prange(samples):
        new_data[i, :, :, 0] = data_to_use[:, i*width_bin:(i+1)*width_bin]
    
    return new_data

@jit(nopython=True, parallel=True)
def combine_cadence(A1, A2, A3, B, C, D) -> np.ndarray:
    """
    FIXED: No normalization here since it's done before injection
    """
    samples = A1.shape[0]
    time_bins = A1.shape[1]
    freq_bins = A1.shape[2]
    
    data = np.zeros((samples, 6, time_bins, freq_bins, 1))
    
    for i in prange(samples):
        # Just combine - no normalization needed
        data[i, 0, :, :, :] = A1[i, :, :, :]
        data[i, 1, :, :, :] = B[i, :, :, :]
        data[i, 2, :, :, :] = A2[i, :, :, :]
        data[i, 3, :, :, :] = C[i, :, :, :]
        data[i, 4, :, :, :] = A3[i, :, :, :]
        data[i, 5, :, :, :] = D[i, :, :, :]
    
    return data

def resize_par(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Resize data in parallel
    Used before feeding to neural network
    
    Args:
        data: Input data (batch, 6, time, freq)
        factor: Downsampling factor (8 in paper)
        
    Returns:
        Downsampled data
    """
    batch, n_obs, time, freq = data.shape
    test = np.zeros((batch, n_obs, time, freq // factor))
    
    for i in range(6):
        test[:, i, :, :] = downscale_local_mean(data[:, i, :, :], (1, 1, factor))
    
    return test

@jit(nopython=True, parallel=True)
def combine_for_nn(data: np.ndarray) -> np.ndarray:
    """
    Combine batch and observation dimensions for neural network input
    
    Args:
        data: Input (batch, 6, time, freq, 1)
        
    Returns:
        Combined (batch*6, time, freq, 1)
    """
    batch = data.shape[0]
    n_obs = data.shape[1]
    new_data = np.zeros((batch * n_obs, data.shape[2], data.shape[3], data.shape[4]))
    
    for i in prange(batch):
        new_data[i*n_obs:(i+1)*n_obs, :, :, :] = data[i, :, :, :, :]
    
    return new_data

class DataPreprocessor:
    """Main preprocessing class"""
    
    def __init__(self, config):
        self.config = config
        self.width_bin = config.data.width_bin
        self.downsample_factor = config.data.downsample_factor
        self.final_freq_bins = self.width_bin // self.downsample_factor 
        
    def process_single_observation(self, obs_data: np.ndarray) -> np.ndarray:
        """
        Process a single raw observation WITHOUT normalization
        Normalization happens after cadence combination
        
        Args:
            obs_data: Raw observation (16, total_freq) or (16, 2, total_freq)
            
        Returns:
            Downsampled snippets (n_snippets, 16, 512)
        """
        # Shape into snippets
        snippets = shaping_data_dynamic(obs_data, self.width_bin)
        
        # Downsample
        downsampled = downscale_local_mean(
            snippets, 
            (1, 1, self.downsample_factor, 1)
        )
        
        return downsampled
    
    def preprocess_cadence(self, observations: List[np.ndarray], 
                           use_overlap: bool = False) -> np.ndarray:
        """
        Preprocess a full cadence
        
        Args:
            observations: List of 6 observation arrays
            use_overlap: Whether to use overlapping windows (for inference only)
            
        Returns:
            Preprocessed cadence data (num_snippets, 6, 16, 512)
        """
        if len(observations) != 6:
            raise ValueError(f"Expected 6 observations, got {len(observations)}")
        
        # Process each observation (downsample but don't normalize yet)
        processed_obs = []
        for obs in observations:
            processed = self.process_single_observation(obs)
            processed_obs.append(processed)
        
        # Get ON and OFF observations
        A1, B, A2, C, A3, D = processed_obs
        
        # Combine and normalize using author's exact function
        cadence_data = combine_cadence(A1, A2, A3, B, C, D)
        
        # Remove channel dimension for compatibility
        cadence_data = cadence_data[:, :, :, :, 0]
        
        return cadence_data
    
    def prepare_for_vae(self, cadence_data: np.ndarray) -> np.ndarray:
        """
        Prepare cadence data for VAE input
        Model expects (batch*6, 16, 512, 1)
        
        Args:
            cadence_data: Cadence data (batch, 6, 16, 512)
            
        Returns:
            VAE-ready data (batch*6, 16, 512, 1)
        """
        # Add channel dimension
        if len(cadence_data.shape) == 4:
            cadence_data = cadence_data[..., np.newaxis]
        
        # Combine batch and observation dimensions
        return combine_for_nn(cadence_data)
    
    def prepare_batch_for_training(self, data_dict: Dict[str, np.ndarray]) -> Tuple:
        """
        Prepare training batch in format expected by VAE
        NO additional normalization - already done in combine_cadence
        
        Args:
            data_dict: Dictionary with 'concatenated', 'true', 'false' keys
            
        Returns:
            ((concatenated, true, false), target) tuple for training
        """
        concatenated = data_dict['concatenated'].astype(np.float32)
        true_data = data_dict['true'].astype(np.float32)
        false_data = data_dict['false'].astype(np.float32)
        
        # Data is already normalized from data generation
        # Just return in the format expected by the model
        return (concatenated, true_data, false_data), concatenated
