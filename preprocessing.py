"""
Data preprocessing for SETI ML Pipeline
FIXED: Preserve 4096 frequency bins as per paper specification
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing and preparation for SETI models"""
    
    def __init__(self, config):
        self.config = config
        self.freq_bins = config.data.width_bin  # 4096
        self.time_bins = config.data.time_bins  # 16
        self.num_observations = config.data.num_observations  # 6
        
        # REMOVED: No downsampling to preserve paper's frequency resolution
        logger.info(f"Preprocessor initialized: {self.freq_bins} freq bins, {self.time_bins} time bins")
        logger.info("NO frequency downsampling - preserving paper's 4096 frequency resolution")
    
    def downsample_frequency(self, data: np.ndarray) -> np.ndarray:
        """
        CHANGED: No-op function - preserve original frequency resolution
        
        Args:
            data: Input data of shape (..., time_bins, freq_bins)
            
        Returns:
            Same data unchanged to preserve 4096 frequency bins
        """
        logger.debug(f"Input shape: {data.shape}")
        
        # Verify we have the expected frequency dimension
        if data.shape[-1] != self.freq_bins:
            logger.warning(f"Expected {self.freq_bins} frequency bins, got {data.shape[-1]}")
            
        # NO DOWNSAMPLING - return data as-is
        logger.debug(f"Output shape: {data.shape} (no downsampling applied)")
        return data
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range for neural network training
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data in range [0, 1]
        """
        # Compute statistics along appropriate axes
        if len(data.shape) == 4:  # (samples, observations, time, freq)
            # Normalize each sample independently
            data_min = np.min(data, axis=(1, 2, 3), keepdims=True)
            data_max = np.max(data, axis=(1, 2, 3), keepdims=True)
        elif len(data.shape) == 3:  # (samples, time, freq)
            # Normalize each sample independently
            data_min = np.min(data, axis=(1, 2), keepdims=True)
            data_max = np.max(data, axis=(1, 2), keepdims=True)
        else:
            # Global normalization
            data_min = np.min(data)
            data_max = np.max(data)
        
        # Avoid division by zero
        data_range = data_max - data_min
        data_range = np.where(data_range == 0, 1, data_range)
        
        normalized = (data - data_min) / data_range
        
        logger.debug(f"Normalized data: min={np.min(normalized):.4f}, max={np.max(normalized):.4f}")
        return normalized.astype(np.float32)
    
    def prepare_batch(self, cadence_data: np.ndarray) -> np.ndarray:
        """
        Prepare cadence data for VAE training
        
        Args:
            cadence_data: Shape (n_samples, 6, 16, 4096) - cadences with 6 observations each
            
        Returns:
            Prepared data: Shape (n_samples*6, 16, 4096, 1) - flattened for VAE input
        """
        logger.debug(f"Preparing batch with input shape: {cadence_data.shape}")
        
        n_samples = cadence_data.shape[0]
        
        # Verify expected shape
        expected_shape = (n_samples, self.num_observations, self.time_bins, self.freq_bins)
        if cadence_data.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {cadence_data.shape}")
        
        # Normalize each cadence
        normalized_data = self.normalize_data(cadence_data)
        
        # Reshape to (n_samples*6, time_bins, freq_bins, 1) for VAE
        flattened = normalized_data.reshape(-1, self.time_bins, self.freq_bins)
        
        # Add channel dimension for CNN
        prepared = np.expand_dims(flattened, axis=-1)
        
        logger.debug(f"Prepared batch output shape: {prepared.shape}")
        logger.info(f"Batch prepared: {n_samples} cadences -> {prepared.shape[0]} individual observations")
        
        return prepared.astype(np.float32)
    
    def prepare_cadence_for_clustering(self, cadence_data: np.ndarray) -> np.ndarray:
        """
        Prepare cadence data for clustering loss computation
        Keeps cadence structure intact
        
        Args:
            cadence_data: Shape (n_samples, 6, 16, 4096)
            
        Returns:
            Normalized data: Shape (n_samples, 6, 16, 4096)
        """
        logger.debug(f"Preparing cadence for clustering: {cadence_data.shape}")
        
        # Normalize while preserving cadence structure
        normalized = self.normalize_data(cadence_data)
        
        logger.debug(f"Clustering data prepared: {normalized.shape}")
        return normalized.astype(np.float32)
    
    def concatenate_observations(self, cadence_data: np.ndarray) -> np.ndarray:
        """
        Concatenate the 6 observations of a cadence along frequency axis
        This is mentioned in the paper for processing
        
        Args:
            cadence_data: Shape (n_samples, 6, 16, 4096)
            
        Returns:
            Concatenated data: Shape (n_samples, 16, 24576) where 24576 = 6 * 4096
        """
        logger.debug(f"Concatenating observations: {cadence_data.shape}")
        
        n_samples = cadence_data.shape[0]
        
        # Concatenate along frequency axis
        concatenated = cadence_data.reshape(n_samples, self.time_bins, -1)
        
        logger.debug(f"Concatenated shape: {concatenated.shape}")
        return concatenated.astype(np.float32)
    
    def validate_dimensions(self, data: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
        """
        Validate that data has expected dimensions
        
        Args:
            data: Input data array
            expected_shape: Expected shape (can use -1 for flexible dimensions)
            
        Returns:
            True if dimensions match, False otherwise
        """
        if len(data.shape) != len(expected_shape):
            logger.error(f"Dimension mismatch: expected {len(expected_shape)} dims, got {len(data.shape)}")
            return False
        
        for i, (actual, expected) in enumerate(zip(data.shape, expected_shape)):
            if expected != -1 and actual != expected:
                logger.error(f"Shape mismatch at dim {i}: expected {expected}, got {actual}")
                return False
        
        logger.debug(f"Dimension validation passed: {data.shape}")
        return True
    
    def create_overlapping_snippets(self, observation: np.ndarray, 
                                  snippet_size: int = 4096, 
                                  overlap: float = 0.5) -> np.ndarray:
        """
        Create overlapping frequency snippets as described in paper
        
        Args:
            observation: Shape (time_bins, full_freq_bins)
            snippet_size: Size of frequency snippet (4096)
            overlap: Overlap fraction (0.5 = 50% overlap)
            
        Returns:
            Snippets: Shape (n_snippets, time_bins, snippet_size)
        """
        time_bins, full_freq_bins = observation.shape
        step_size = int(snippet_size * (1 - overlap))
        
        snippets = []
        start = 0
        
        while start + snippet_size <= full_freq_bins:
            snippet = observation[:, start:start + snippet_size]
            snippets.append(snippet)
            start += step_size
        
        if len(snippets) == 0:
            # If observation is smaller than snippet_size, pad it
            if full_freq_bins < snippet_size:
                padding = snippet_size - full_freq_bins
                padded = np.pad(observation, ((0, 0), (0, padding)), mode='constant', constant_values=0)
                snippets.append(padded)
            else:
                # Take the last possible snippet
                snippet = observation[:, -snippet_size:]
                snippets.append(snippet)
        
        result = np.array(snippets)
        logger.debug(f"Created {len(snippets)} overlapping snippets from {observation.shape}")
        
        return result
