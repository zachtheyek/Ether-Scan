#!/usr/bin/env python3
"""
Emergency debug script to identify exact source of loss explosion
This will run just a few training steps with debug output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from data_generation import DataGenerator, create_mixed_training_batch
from preprocessing import DataPreprocessor
from models.vae import create_vae_model
import logging

# Minimal config for testing
class TestConfig:
    class model:
        latent_dim = 8
        dense_layer_size = 512
        kernel_size = (3, 3)
        alpha = 1.0
        beta = 0.5
        gamma = 0.0
        learning_rate = 0.0001
        
    class training:
        batch_size = 32  # Small batch for debugging
        
def emergency_debug_test():
    """Run minimal training steps with debug output to find explosion source"""
    
    print("=" * 80)
    print("EMERGENCY LOSS DEBUG - Finding explosion source")
    print("=" * 80)
    
    # Create synthetic background data
    background_data = np.random.randn(100, 6, 16, 512).astype(np.float32) * 0.1
    
    config = TestConfig()
    data_generator = DataGenerator(config, background_data)
    
    # Use single GPU to avoid distributed complexity for debugging
    with tf.device('/GPU:0'):
        print("Creating VAE model...")
        vae = create_vae_model(config)
        
        print("Running debug training steps...")
        
        for step in range(10):  # Run 10 steps to see progression
            print(f"\n--- DEBUG STEP {step + 1} ---")
            
            try:
                # Create batch
                combined_data, true_data, false_data = create_mixed_training_batch(
                    data_generator, config.training.batch_size
                )
                
                # Prepare data
                inputs = (combined_data.astype(np.float32), 
                         true_data.astype(np.float32), 
                         false_data.astype(np.float32))
                target = combined_data.astype(np.float32)
                
                # Single training step with debug output
                result = vae.train_on_batch(inputs, target)
                
                print(f"Step {step + 1} results:")
                for key, value in result.items():
                    print(f"  {key}: {value}")
                
                # Check for explosion
                if any(abs(float(v)) > 1000 for v in result.values()):
                    print(f"*** EXPLOSION DETECTED AT STEP {step + 1} ***")
                    break
                    
            except Exception as e:
                print(f"*** ERROR AT STEP {step + 1}: {e} ***")
                break
                
        print("\n" + "=" * 80)
        print("Debug complete. Check output above for loss progression.")
        print("=" * 80)

if __name__ == "__main__":
    # Reduce TF logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    emergency_debug_test()