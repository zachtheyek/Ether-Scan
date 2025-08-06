"""
Splits the test background into distinct cadences for inference
"""

import numpy as np
import os
import sys
import argparse
import logging

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_test_cadences(config: Config, max_cadences: int = 10):
    """
    Prepare test data cadences for inference
    
    Args:
        config: Configuration object
        max_cadences: Maximum number of cadences to prepare
    """
    # Iterate through test files
    for test_file in config.data.test_files:
        test_path = config.get_test_file_path(test_file)
        
        if not os.path.exists(test_path):
            logger.warning(f"Test file not found: {test_path}")
            continue
            
        logger.info(f"Processing {test_file}")
        
        # Load test data
        test_data = np.load(test_path)
        logger.info(f"  Shape: {test_data.shape}")
        
        # Apply subset if configured
        start, end = config.get_file_subset(test_file)
        if start is not None or end is not None:
            test_data = test_data[start:end]
            logger.info(f"  Applied subset [{start}:{end}], new shape {test_data.shape}")
        
        # Process cadences
        n_cadences = min(max_cadences, test_data.shape[0])
        
        # Create output directory for this test file
        output_dir = os.path.join(
            config.data_path, 
            'testing', 
            f'prepared_{os.path.splitext(test_file)[0]}'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each cadence as separate observation files
        for i in range(n_cadences):
            cadence = test_data[i]  # Shape: (6, 16, 4096)
            
            # Save each observation
            for j in range(6):
                obs_file = os.path.join(output_dir, f'cadence_{i:04d}_obs_{j}.npy')
                np.save(obs_file, cadence[j])
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1} cadences")
        
        logger.info(f"  Saved {n_cadences} cadences to {output_dir}")
        
        # Also create a metadata file
        metadata = {
            'source_file': test_file,
            'n_cadences': n_cadences,
            'cadence_shape': list(cadence.shape),
            'output_dir': output_dir
        }
        
        import json
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Saved metadata to {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare test data for inference')
    parser.add_argument('--max-cadences', type=int, default=10,
                       help='Maximum number of cadences to prepare per file')
    parser.add_argument('--config-file', type=str,
                       help='Path to custom configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override with custom config if provided
    if args.config_file:
        import json
        with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
            # Apply custom settings
            # This is a simple example - you might want more sophisticated merging
            if 'data' in custom_config:
                for key, value in custom_config['data'].items():
                    if hasattr(config.data, key):
                        setattr(config.data, key, value)
    
    # Prepare test cadences
    prepare_test_cadences(config, args.max_cadences)

if __name__ == '__main__':
    main()
