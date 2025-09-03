"""
Data preprocessing module for SETI ML Pipeline
Fixed to match author's exact preprocessing flow
"""

import numpy as np
from numba import jit, prange, njit
from skimage.transform import downscale_local_mean
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

@njit(nopython=True)
def pre_proc(data: np.ndarray) -> np.ndarray:
    """
    Apply log normalization to data following author's exact approach
    This matches the author's pre_proc function exactly
    
    Args:
        data: Input array
        
    Returns:
        Normalized array between 0 and 1
    """
    # Author's exact normalization sequence
    data = np.log(data)
    data = data - data.min()
    data = data / data.max()
    return data

@jit(parallel=True)
def shaping_data_dynamic(data: np.ndarray, width_bin: int = 4096) -> np.ndarray:
    """
    Reshape raw observation data into snippets
    Matches author's shaping_data function from preprocess_dynamic.py
    
    Args:
        data: Raw observation data (time_bins, polarization, frequency_channels)
              Expected shape: (16, 1, total_freq_channels) for single pol
        width_bin: Number of frequency bins per snippet (4096 as per paper)
        
    Returns:
        Reshaped data (num_snippets, time_bins, freq_bins, 1)
    """
    # Handle both single and dual polarization
    if len(data.shape) == 3:
        time_bins, n_pol, total_freq = data.shape
    else:
        # Single polarization - add dummy dimension
        time_bins, total_freq = data.shape
        data = data.reshape(time_bins, 1, total_freq)
        n_pol = 1
    
    samples = total_freq // width_bin
    new_data = np.zeros((samples, time_bins, width_bin, 1))
    
    for i in prange(samples):
        # Use first polarization only
        new_data[i, :, :, 0] = data[:, 0, i*width_bin:(i+1)*width_bin]
    
    return new_data

@jit(parallel=True)
def combine_cadence(A1, A2, A3, B, C, D) -> np.ndarray:
    """
    Combine 6 observations into cadence array and normalize
    Matches author's combine_cadence function exactly
    
    Args:
        A1, A2, A3: ON observations 
        B, C, D: OFF observations
        
    Returns:
        Combined and normalized array (samples, 6, 16, 512, 1)
    """
    samples = A1.shape[0]
    time_bins = A1.shape[1]
    freq_bins = A1.shape[2]
    
    data = np.zeros((samples, 6, time_bins, freq_bins, 1))
    
    for i in prange(samples):
        data[i, 0, :, :, :] = A1[i, :, :, :]
        data[i, 1, :, :, :] = B[i, :, :, :]
        data[i, 2, :, :, :] = A2[i, :, :, :]
        data[i, 3, :, :, :] = C[i, :, :, :]
        data[i, 4, :, :, :] = A3[i, :, :, :]
        data[i, 5, :, :, :] = D[i, :, :, :]
        
        # Critical: Normalize AFTER combining, as author does
        data[i, :, :, :, :] = pre_proc(data[i, :, :, :, :])
    
    return data

def get_data_dynamic(cadence: List[str], start_freq: float, end_freq: float,
                    width_bin: int = 4096, downsample_factor: int = 8):
    """
    Load and preprocess a cadence following author's get_data function
    """
    from blimpy import Waterfall
    import time
    
    start_time = time.time()
    
    # Load first observation to get dimensions
    A1 = Waterfall(cadence[0], f_start=start_freq, f_stop=end_freq).data
    A1_shaped = shaping_data_dynamic(A1[:, 0, :], width_bin)  # Take first pol
    
    # Downsample
    A1_down = downscale_local_mean(A1_shaped, (1, 1, downsample_factor, 1))
    
    # Load and process other observations
    B = shaping_data_dynamic(Waterfall(cadence[1], f_start=start_freq, f_stop=end_freq).data[:, 0, :], width_bin)
    B = downscale_local_mean(B, (1, 1, downsample_factor, 1))
    
    A2 = shaping_data_dynamic(Waterfall(cadence[2], f_start=start_freq, f_stop=end_freq).data[:, 0, :], width_bin)
    A2 = downscale_local_mean(A2, (1, 1, downsample_factor, 1))
    
    C = shaping_data_dynamic(Waterfall(cadence[3], f_start=start_freq, f_stop=end_freq).data[:, 0, :], width_bin)
    C = downscale_local_mean(C, (1, 1, downsample_factor, 1))
    
    A3 = shaping_data_dynamic(Waterfall(cadence[4], f_start=start_freq, f_stop=end_freq).data[:, 0, :], width_bin)
    A3 = downscale_local_mean(A3, (1, 1, downsample_factor, 1))
    
    D = shaping_data_dynamic(Waterfall(cadence[5], f_start=start_freq, f_stop=end_freq).data[:, 0, :], width_bin)
    D = downscale_local_mean(D, (1, 1, downsample_factor, 1))
    
    # Combine and normalize
    data = combine_cadence(A1_down, A2, A3, B, C, D)
    
    logger.info(f"Data Load Execution Time: {time.time() - start_time:.2f}s")
    
    return data

def resize_par(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Resize data in parallel, matching author's resize_par function
    
    Args:
        data: Input data (batch, 6, time, freq)
        factor: Downsampling factor
        
    Returns:
        Downsampled data
    """
    batch, n_obs, time, freq = data.shape
    test = np.zeros((batch, n_obs, time, freq // factor))
    
    for i in range(6):
        test[:, i, :, :] = downscale_local_mean(data[:, i, :, :], (1, 1, factor))
    
    return test

@jit(parallel=True)
def load_data_ED(data: np.ndarray) -> np.ndarray:
    """
    Load and normalize data for encoder-decoder
    Matches author's load_data_ED function from data_generation.py
    
    Args:
        data: Input data (batch, 6, time, freq)
        
    Returns:
        Normalized data (batch, 6, time, freq, 1)
    """
    batch, n_obs, time, freq = data.shape
    data_transform = np.zeros((batch, n_obs, time, freq, 1))
    
    for i in prange(batch):
        data_transform[i, :, :, :, 0] = pre_proc(data[i, :, :, :])
    
    return data_transform

@jit(parallel=True)
def combine(data: np.ndarray) -> np.ndarray:
    """
    Combine batch and observation dimensions for neural network input
    Matches author's combine function exactly
    
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
    """Main preprocessing class matching author's approach"""
    
    def __init__(self, config):
        self.config = config
        self.width_bin = config.data.width_bin  # 4096
        self.downsample_factor = config.data.downsample_factor  # 8
        
    def preprocess_observation(self, obs_data: np.ndarray) -> np.ndarray:
        """
        Preprocess single observation
        
        Args:
            obs_data: Raw observation (16, total_freq) or (16, 2, total_freq)
            
        Returns:
            Processed snippets (n_snippets, 16, 512)
        """
        # Ensure 3D shape
        if len(obs_data.shape) == 2:
            obs_data = obs_data[:, np.newaxis, :]
        
        # Shape into snippets
        snippets = shaping_data_dynamic(obs_data, self.width_bin)
        
        # Downsample
        downsampled = downscale_local_mean(
            snippets, 
            (1, 1, self.downsample_factor, 1)
        )
        
        # Remove channel dimension for now
        downsampled = downsampled[:, :, :, 0]
        
        return downsampled
    
    def preprocess_cadence(self, observations: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a full cadence of observations
        
        Args:
            observations: List of 6 observation arrays
            
        Returns:
            Preprocessed cadence data (num_snippets, 6, 16, 512)
        """
        if len(observations) != 6:
            raise ValueError(f"Expected 6 observations, got {len(observations)}")
        
        # Process each observation
        processed_obs = []
        for obs in observations:
            processed = self.preprocess_observation(obs)
            processed_obs.append(processed)
        
        # Get number of snippets (should be same for all)
        n_snippets = processed_obs[0].shape[0]
        
        # Stack into cadence format
        cadence_data = np.zeros((n_snippets, 6, 16, self.width_bin // self.downsample_factor))
        
        for i in range(6):
            cadence_data[:, i, :, :] = processed_obs[i]
        
        # Apply normalization to entire cadence
        for i in range(n_snippets):
            cadence_data[i] = pre_proc(cadence_data[i])
        
        return cadence_data
    
    def prepare_for_vae(self, cadence_data: np.ndarray) -> np.ndarray:
        """
        Prepare cadence data for VAE input
        
        Args:
            cadence_data: Cadence data (batch, 6, 16, 512) or with channel
            
        Returns:
            VAE-ready data (batch*6, 16, 512, 1)
        """
        # Add channel dimension if needed
        if len(cadence_data.shape) == 4:
            cadence_data = cadence_data[..., np.newaxis]
        
        # Combine batch and observation dimensions
        return combine(cadence_data)
    
    def prepare_batch_for_training(self, data_dict: Dict[str, np.ndarray]) -> Tuple:
        """
        Prepare training batch in format expected by VAE
        
        Args:
            data_dict: Dictionary with 'concatenated', 'true', 'false' keys
            
        Returns:
            ((concatenated, true, false), target) tuple
        """
        # Ensure all data is normalized
        concatenated = data_dict['concatenated']
        true_data = data_dict['true']
        false_data = data_dict['false']
        
        # Add channel dimension if needed
        if len(concatenated.shape) == 4:
            concatenated = load_data_ED(concatenated)
            true_data = load_data_ED(true_data)
            false_data = load_data_ED(false_data)
        
        # Remove channel dimension for training (model adds it back)
        concatenated = concatenated[:, :, :, :, 0]
        true_data = true_data[:, :, :, :, 0]
        false_data = false_data[:, :, :, :, 0]
        
        return (concatenated, true_data, false_data), concatenated
