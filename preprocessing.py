"""
Data preprocessing module for SETI ML Pipeline
Handles data loading, normalization, and shaping
"""

import numpy as np
from numba import jit, prange, njit
from skimage.transform import downscale_local_mean
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

@njit(nopython=True)
def normalize_log(data: np.ndarray) -> np.ndarray:
    """
    Apply log normalization to data
    
    Args:
        data: Input array
        
    Returns:
        Normalized array between 0 and 1
    """
    # Add small epsilon to avoid log(0)
    data = np.log(data + 1e-10)
    data = data - data.min()
    if data.max() > 0:
        data = data / data.max()
    return data

@jit(parallel=True)
def shape_observation_data(data: np.ndarray, width_bin: int = 4096) -> np.ndarray:
    """
    Reshape raw observation data into snippets
    
    Args:
        data: Raw observation data (time, polarization, frequency)
        width_bin: Number of frequency bins per snippet
        
    Returns:
        Reshaped data (num_snippets, time_bins, freq_bins, channels)
    """
    samples = data.shape[2] // width_bin
    new_data = np.zeros((samples, 16, width_bin, 1))
    
    for i in prange(samples):
        new_data[i, :, :, 0] = data[:, 0, i*width_bin:(i+1)*width_bin]
    
    return new_data

@jit(parallel=True)
def combine_cadence_observations(A1, A2, A3, B, C, D, width_bin: int, factor: int):
    """
    Combine 6 observations into a single cadence array
    
    Args:
        A1, A2, A3: ON observations
        B, C, D: OFF observations
        width_bin: Frequency bins
        factor: Downsampling factor
        
    Returns:
        Combined array (samples, 6, time_bins, freq_bins, 1)
    """
    samples = A1.shape[0]
    data = np.zeros((samples, 6, 16, width_bin//factor, 1))
    
    for i in prange(samples):
        # ON observations
        data[i, 0, :, :, :] = A1[i, :, :, :]
        data[i, 2, :, :, :] = A2[i, :, :, :]
        data[i, 4, :, :, :] = A3[i, :, :, :]
        # OFF observations
        data[i, 1, :, :, :] = B[i, :, :, :]
        data[i, 3, :, :, :] = C[i, :, :, :]
        data[i, 5, :, :, :] = D[i, :, :, :]
        
        # Normalize each cadence
        data[i, :, :, :, :] = normalize_log(data[i, :, :, :, :])
    
    return data

def downsample_frequency(data: np.ndarray, factor: int = 8) -> np.ndarray:
    """
    Downsample data in frequency dimension
    
    Args:
        data: Input data
        factor: Downsampling factor
        
    Returns:
        Downsampled data
    """
    if len(data.shape) == 4:
        # Process each observation separately for 4D data (batch, obs, time, freq)
        output_shape = list(data.shape)
        output_shape[-1] = output_shape[-1] // factor
        output = np.zeros(output_shape)
        
        for i in range(data.shape[1]):
            output[:, i, :, :] = downscale_local_mean(
                data[:, i, :, :], (1, 1, factor)
            )
    else:
        # Handle 3D data (batch, time, freq) directly
        output = downscale_local_mean(data, (1, 1, factor))
    
    return output

@jit(parallel=True)
def prepare_for_model(data: np.ndarray) -> np.ndarray:
    """
    Prepare data for model input by combining observations
    
    Args:
        data: Cadence data (batch, 6, time, freq, 1)
        
    Returns:
        Combined data (batch*6, time, freq, 1)
    """
    batch_size = data.shape[0]
    num_obs = data.shape[1]
    new_data = np.zeros((batch_size * num_obs, data.shape[2], data.shape[3], data.shape[4]))
    
    for i in prange(batch_size):
        new_data[i*num_obs:(i+1)*num_obs, :, :, :] = data[i, :, :, :, :]
    
    return new_data

class DataPreprocessor:
    """Main preprocessing class"""
    
    def __init__(self, config):
        self.config = config
        self.width_bin = config.data.width_bin
        self.downsample_factor = config.data.downsample_factor
        
    def preprocess_cadence(self, observations: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a full cadence of observations
        
        Args:
            observations: List of 6 observation arrays
            
        Returns:
            Preprocessed cadence data
        """
        if len(observations) != 6:
            raise ValueError(f"Expected 6 observations, got {len(observations)}")
        
        # Shape each observation
        shaped_obs = []
        for obs in observations:
            shaped = shape_observation_data(obs, self.width_bin)
            downsampled = downscale_local_mean(shaped, (1, 1, self.downsample_factor, 1))
            shaped_obs.append(downsampled)
        
        # Combine into cadence
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
            cadences: Batch of cadence data
            
        Returns:
            Data ready for model input
        """
        return prepare_for_model(cadences)
    
    def downsample_frequency(self, data: np.ndarray, factor: int = 8) -> np.ndarray:
        """
        Downsample data in frequency dimension
        
        Args:
            data: Input data
            factor: Downsampling factor
            
        Returns:
            Downsampled data
        """
        return downsample_frequency(data, factor)
    
    def extract_snippets_with_overlap(self, data: np.ndarray, overlap: float = 0.5) -> List[np.ndarray]:
        """
        Extract overlapping snippets from continuous data
        
        Args:
            data: Continuous observation data
            overlap: Overlap factor (0.5 = 50% overlap)
            
        Returns:
            List of snippet arrays
        """
        snippet_size = self.width_bin
        step_size = int(snippet_size * (1 - overlap))
        
        snippets = []
        for start in range(0, data.shape[-1] - snippet_size + 1, step_size):
            snippet = data[..., start:start + snippet_size]
            snippets.append(snippet)
        
        return snippets
