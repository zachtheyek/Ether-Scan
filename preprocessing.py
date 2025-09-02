"""
Data preprocessing module for SETI ML Pipeline
Handles data loading, normalization, and shaping
Fixed to match paper dimensions and processing flow
"""

import numpy as np
from numba import jit, prange, njit
from skimage.transform import downscale_local_mean
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

def normalize_log(data: np.ndarray, already_normalized: bool = False) -> np.ndarray:
    """
    Apply log normalization to data as per paper with safety checks
    
    Args:
        data: Input array
        
    Returns:
        Normalized array between 0 and 1 (guaranteed float32)
    """
    if already_normalized:
        return data # Skip if already normalized 

    # Convert to float32 for consistent precision
    data = data.astype(np.float32)

    # NOTE: not sure if this is needed. leave in for now
    # Handle the massive raw values from verify_data.log
    if np.max(data) > 1e6:  # Likely raw telescope data
        # Scale down first to reasonable range
        data = data / np.median(data)

    # Ensure positive values
    data_min = np.min(data)
    if data_min <= 0:
        data = data - data_min + 1e-8
    
    # Single log transform
    data_log = np.log(data + 1e-8)
    
    # Normalize to [0, 1]
    log_min, log_max = np.min(data_log), np.max(data_log)
    if (log_max - log_min) > 1e-10:
        return (data_log - log_min) / (log_max - log_min)
    else:
        return np.ones_like(data) * 0.5

# Removed @jit decorator to avoid Numba warnings and improve stability
def shape_observation_data(data: np.ndarray, width_bin: int = 4096) -> np.ndarray:
    """
    Reshape raw observation data into snippets
    Paper: Extract 4096-channel snippets from continuous observation
    
    Args:
        data: Raw observation data (time_bins, polarization, frequency_channels)
              Expected shape: (16, 2, total_freq_channels)
        width_bin: Number of frequency bins per snippet (4096 as per paper)
        
    Returns:
        Reshaped data (num_snippets, time_bins, freq_bins)
        Note: No channel dimension here - added later for model input
    """
    time_bins, n_pol, total_freq = data.shape
    samples = total_freq // width_bin
    
    # Output shape: (samples, 16, 4096) - no channel dimension yet
    new_data = np.zeros((samples, time_bins, width_bin))
    
    for i in range(samples):
        # Extract snippet from first polarization only (as per paper)
        new_data[i, :, :] = data[:, 0, i*width_bin:(i+1)*width_bin]
    
    return new_data

def combine_cadence_observations(A1, A2, A3, B, C, D, width_bin: int = 4096, factor: int = 8):
    """
    Combine 6 observations into a single cadence array
    Paper: ABACAD or ABABAB pattern, 3 ON and 3 OFF observations
    
    Args:
        A1, A2, A3: ON observations (shape: samples, time, freq)
        B, C, D: OFF observations (shape: samples, time, freq)
        width_bin: Frequency bins (4096)
        factor: Downsampling factor (8)
        
    Returns:
        Combined array (samples, 6, time_bins, freq_bins_downsampled)
    """
    samples = A1.shape[0]
    time_bins = A1.shape[1]  # 16
    freq_bins_downsampled = width_bin // factor  # 512
    
    # Shape: (samples, 6, 16, 512)
    data = np.zeros((samples, 6, time_bins, freq_bins_downsampled))
    
    # Downsample each observation
    for i in range(samples):
        # ON observations (indices 0, 2, 4)
        data[i, 0, :, :] = downscale_local_mean(A1[i], (1, factor))
        data[i, 2, :, :] = downscale_local_mean(A2[i], (1, factor))
        data[i, 4, :, :] = downscale_local_mean(A3[i], (1, factor))
        
        # OFF observations (indices 1, 3, 5)
        data[i, 1, :, :] = downscale_local_mean(B[i], (1, factor))
        data[i, 3, :, :] = downscale_local_mean(C[i], (1, factor))
        data[i, 5, :, :] = downscale_local_mean(D[i], (1, factor))
        
        # Normalize each cadence using log normalization
        for j in range(6):
            data[i, j, :, :] = normalize_log(data[i, j, :, :])
    
    return data

def downsample_frequency(data: np.ndarray, factor: int = 8) -> np.ndarray:
    """
    Downsample data in frequency dimension
    Paper: Downsample by factor of 8 to get 512 frequency bins
    
    Args:
        data: Input data (various shapes)
        factor: Downsampling factor (8)
        
    Returns:
        Downsampled data
    """
    if len(data.shape) == 3:
        # Shape: (batch, time, freq) -> (batch, time, freq//factor)
        return downscale_local_mean(data, (1, 1, factor))
    elif len(data.shape) == 4:
        # Shape: (batch, obs, time, freq) -> (batch, obs, time, freq//factor)
        output_shape = list(data.shape)
        output_shape[-1] = output_shape[-1] // factor
        output = np.zeros(output_shape)
        
        for i in range(data.shape[1]):
            output[:, i, :, :] = downscale_local_mean(data[:, i, :, :], (1, 1, factor))
        return output
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

def prepare_for_model(data: np.ndarray) -> np.ndarray:
    """
    Prepare data for model input by adding channel dimension and flattening batch
    Paper: Input to VAE is (16, 512, 1) per observation
    
    Args:
        data: Cadence data (batch, 6, time, freq)
        
    Returns:
        Combined data (batch*6, time, freq, 1)
    """
    batch_size = data.shape[0]
    num_obs = data.shape[1]  # 6
    time_bins = data.shape[2]  # 16
    freq_bins = data.shape[3]  # 512
    
    # Flatten batch and observations, add channel dimension
    # Output: (batch*6, 16, 512, 1)
    new_data = np.zeros((batch_size * num_obs, time_bins, freq_bins, 1))
    
    for i in range(batch_size):
        for j in range(num_obs):
            new_data[i*num_obs + j, :, :, 0] = data[i, j, :, :]
    
    return new_data

class DataPreprocessor:
    """Main preprocessing class"""
    
    def __init__(self, config):
        self.config = config
        self.width_bin = config.data.width_bin  # 4096
        self.downsample_factor = config.data.downsample_factor  # 8
        
    def preprocess_cadence(self, observations: List[np.ndarray], 
                          use_overlap: bool = True) -> np.ndarray:
        """
        Preprocess a full cadence of observations
        
        Args:
            observations: List of 6 observation arrays, each (16, 2, freq_channels)
            use_overlap: Whether to extract overlapping snippets (for inference only)
            
        Returns:
            Preprocessed cadence data (num_snippets, 6, 16, 512)
        """
        if len(observations) != 6:
            raise ValueError(f"Expected 6 observations, got {len(observations)}")
        
        shaped_obs = []
        for obs in observations:
            if use_overlap:
                # Extract with 50% overlap as per paper (for inference only)
                snippets = self.extract_snippets_with_overlap(obs, overlap=0.5)
                shaped = np.array(snippets)
            else:
                # Original non-overlapping extraction (for training)
                # obs shape: (16, 2, total_freq) -> (num_snippets, 16, 4096)
                shaped = shape_observation_data(obs, self.width_bin)
            shaped_obs.append(shaped)
        
        # Combine into cadence with downsampling
        combined = combine_cadence_observations(
            *shaped_obs, 
            self.width_bin, 
            self.downsample_factor
        )
        
        return combined
    
    def prepare_batch(self, cadences: np.ndarray) -> np.ndarray:
        """
        Prepare batch of cadences for model input
        
        Args:
            cadences: Batch of cadence data (batch, 6, 16, 512)
            
        Returns:
            Data ready for model input (batch*6, 16, 512, 1)
        """
        return prepare_for_model(cadences)
    
    def downsample_frequency(self, data: np.ndarray, factor: Optional[int] = None) -> np.ndarray:
        """
        Downsample data in frequency dimension
        
        Args:
            data: Input data
            factor: Downsampling factor (default: 8)
            
        Returns:
            Downsampled data
        """
        if factor is None:
            factor = self.downsample_factor
        return downsample_frequency(data, factor)
    
    def extract_snippets_with_overlap(self, data: np.ndarray, overlap: float = 0.5) -> List[np.ndarray]:
        """
        Extract overlapping snippets from continuous data
        Paper: 50% overlap for better coverage
        
        Args:
            data: Continuous observation data (time, freq)
            overlap: Overlap factor (0.5 = 50% overlap)
            
        Returns:
            List of snippet arrays
        """
        snippet_size = self.width_bin
        step_size = int(snippet_size * (1 - overlap))
        
        snippets = []
        freq_dimension = data.shape[-1]
        
        for start in range(0, freq_dimension - snippet_size + 1, step_size):
            snippet = data[..., start:start + snippet_size]
            snippets.append(snippet)
        
        return snippets
