#!/usr/bin/env python3
"""
Systematic debugging script to isolate the NaN source
Run this to methodically test each component of the pipeline
"""

import sys
import os
sys.path.append('.')

import numpy as np
import tensorflow as tf
from config import Config
from preprocessing import DataPreprocessor, pre_proc
from data_generation import DataGenerator
from models.vae import create_vae_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_for_nans(data, name):
    """Check data for NaN/Inf and print statistics"""
    has_nan = np.any(np.isnan(data))
    has_inf = np.any(np.isinf(data))
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    
    print(f"\n{name} statistics:")
    print(f"  Shape: {data.shape}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"  Mean: {mean_val:.6f}")
    
    if has_nan or has_inf:
        print(f"  ❌ PROBLEM: {name} contains NaN or Inf!")
        return False
    else:
        print(f"  ✅ {name} is clean")
        return True

def test_preprocessing_pipeline():
    """Test preprocessing step by step"""
    print("="*60)
    print("TESTING PREPROCESSING PIPELINE")
    print("="*60)
    
    # Create synthetic test data
    test_data = np.random.randn(16, 4096).astype(np.float32) + 1000.0
    print(f"Generated test data with mean {np.mean(test_data):.2f}")
    
    # Test log normalization step
    print("\n--- Testing pre_proc (log normalization) ---")
    
    # Check input
    check_data_for_nans(test_data, "Input to pre_proc")
    
    # Apply preprocessing
    try:
        processed = pre_proc(test_data)
        clean = check_data_for_nans(processed, "Output of pre_proc")
        if not clean:
            return False
    except Exception as e:
        print(f"❌ pre_proc failed: {e}")
        return False
    
    # Test with edge cases
    print("\n--- Testing edge cases ---")
    
    # Test with very small values
    small_data = np.random.randn(16, 512).astype(np.float32) * 1e-10 + 1e-9
    try:
        processed_small = pre_proc(small_data)
        check_data_for_nans(processed_small, "Small values processed")
    except Exception as e:
        print(f"❌ Small values failed: {e}")
        return False
    
    # Test with negative values (should be handled by pre_proc)
    neg_data = np.random.randn(16, 512).astype(np.float32) - 100.0
    try:
        processed_neg = pre_proc(neg_data)
        check_data_for_nans(processed_neg, "Negative values processed")
    except Exception as e:
        print(f"❌ Negative values failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test synthetic data generation"""
    print("\n" + "="*60)
    print("TESTING DATA GENERATION")
    print("="*60)
    
    config = Config()
    
    # Create clean background data
    background_data = np.random.randn(100, 6, 16, 512).astype(np.float32) + 1000.0
    
    # Normalize each cadence individually
    print("Normalizing background data...")
    for i in range(background_data.shape[0]):
        for j in range(6):
            background_data[i, j] = pre_proc(background_data[i, j])
    
    check_data_for_nans(background_data, "Normalized background data")
    
    # Test data generator
    print("\n--- Testing DataGenerator ---")
    try:
        generator = DataGenerator(config, background_data)
        
        # Generate small batch
        batch_data = generator.generate_training_batch(8)
        
        # Check each component
        for key, data in batch_data.items():
            clean = check_data_for_nans(data, f"Generated {key}")
            if not clean:
                return False
                
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False
    
    return True

def test_model_initialization():
    """Test model creation and initialization"""
    print("\n" + "="*60)
    print("TESTING MODEL INITIALIZATION")
    print("="*60)
    
    config = Config()
    
    try:
        # Create model
        vae = create_vae_model(config)
        
        # Check encoder weights
        print("\n--- Checking encoder weights ---")
        for i, layer in enumerate(vae.encoder.layers):
            if hasattr(layer, 'weights') and layer.weights:
                for j, weight in enumerate(layer.weights):
                    weight_vals = weight.numpy()
                    clean = check_data_for_nans(weight_vals, f"Encoder layer {i} weight {j}")
                    if not clean:
                        return False
        
        # Check decoder weights
        print("\n--- Checking decoder weights ---")
        for i, layer in enumerate(vae.decoder.layers):
            if hasattr(layer, 'weights') and layer.weights:
                for j, weight in enumerate(layer.weights):
                    weight_vals = weight.numpy()
                    clean = check_data_for_nans(weight_vals, f"Decoder layer {i} weight {j}")
                    if not clean:
                        return False
                        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False
    
    return True

def test_forward_pass():
    """Test forward pass with synthetic data"""
    print("\n" + "="*60)
    print("TESTING FORWARD PASS")
    print("="*60)
    
    config = Config()
    
    # Create model
    vae = create_vae_model(config)
    
    # Create clean synthetic data
    batch_size = 4
    test_input = np.random.randn(batch_size, 6, 16, 512).astype(np.float32) * 0.1 + 0.5
    
    # Normalize the test input
    for i in range(batch_size):
        for j in range(6):
            test_input[i, j] = pre_proc(test_input[i, j])
    
    check_data_for_nans(test_input, "Test input")
    
    # Test encoder only
    print("\n--- Testing encoder ---")
    try:
        # Reshape for encoder
        encoder_input = test_input.reshape(batch_size * 6, 16, 512, 1)
        z_mean, z_log_var, z = vae.encoder(encoder_input, training=False)
        
        check_data_for_nans(z_mean.numpy(), "Encoder z_mean")
        check_data_for_nans(z_log_var.numpy(), "Encoder z_log_var") 
        check_data_for_nans(z.numpy(), "Encoder z")
        
    except Exception as e:
        print(f"❌ Encoder forward pass failed: {e}")
        return False
    
    # Test decoder
    print("\n--- Testing decoder ---")
    try:
        reconstruction = vae.decoder(z, training=False)
        check_data_for_nans(reconstruction.numpy(), "Decoder output")
        
    except Exception as e:
        print(f"❌ Decoder forward pass failed: {e}")
        return False
    
    # Test individual loss components
    print("\n--- Testing loss components ---")
    
    # Reconstruction loss
    try:
        reconstruction = reconstruction.numpy().reshape(batch_size, 6, 16, 512)
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(test_input, reconstruction), 
                axis=(1, 2)
            )
        )
        print(f"Reconstruction loss: {recon_loss.numpy():.6f}")
        if np.isnan(recon_loss.numpy()) or np.isinf(recon_loss.numpy()):
            print("❌ Reconstruction loss is NaN/Inf")
            return False
            
    except Exception as e:
        print(f"❌ Reconstruction loss failed: {e}")
        return False
    
    # KL loss
    try:
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        print(f"KL loss: {kl_loss.numpy():.6f}")
        if np.isnan(kl_loss.numpy()) or np.isinf(kl_loss.numpy()):
            print("❌ KL loss is NaN/Inf")
            return False
            
    except Exception as e:
        print(f"❌ KL loss failed: {e}")
        return False
    
    # Clustering losses
    try:
        true_loss = vae.compute_clustering_loss_true(test_input)
        false_loss = vae.compute_clustering_loss_false(test_input)
        
        print(f"True clustering loss: {true_loss.numpy():.6f}")
        print(f"False clustering loss: {false_loss.numpy():.6f}")
        
        if np.isnan(true_loss.numpy()) or np.isinf(true_loss.numpy()):
            print("❌ True clustering loss is NaN/Inf") 
            return False
        if np.isnan(false_loss.numpy()) or np.isinf(false_loss.numpy()):
            print("❌ False clustering loss is NaN/Inf")
            return False
            
    except Exception as e:
        print(f"❌ Clustering loss failed: {e}")
        return False
    
    return True

def test_real_data_pipeline():
    """Test with real training data"""
    print("\n" + "="*60)
    print("TESTING REAL DATA PIPELINE")
    print("="*60)
    
    config = Config()
    
    # Load a small amount of real background data
    try:
        data_file = "/datax/scratch/zachy/data/etherscan/training/real_filtered_LARGE_HIP110750.npy"
        if os.path.exists(data_file):
            print(f"Loading real data from {data_file}")
            real_data = np.load(data_file, mmap_mode='r')
            
            # Take just a few samples
            sample_data = np.array(real_data[0:10])
            print(f"Loaded sample shape: {sample_data.shape}")
            
            check_data_for_nans(sample_data, "Raw real data")
            
            # Apply downsampling and normalization
            from skimage.transform import downscale_local_mean
            processed_samples = []
            
            for i in range(sample_data.shape[0]):
                cadence = sample_data[i]  # Shape: (6, 16, 4096)
                
                # Downsample
                downsampled = np.zeros((6, 16, 512), dtype=np.float32)
                for obs_idx in range(6):
                    downsampled[obs_idx] = downscale_local_mean(
                        cadence[obs_idx], (1, 8)
                    ).astype(np.float32)
                
                # Normalize
                for obs_idx in range(6):
                    downsampled[obs_idx] = pre_proc(downsampled[obs_idx])
                
                processed_samples.append(downsampled)
            
            processed_array = np.array(processed_samples)
            check_data_for_nans(processed_array, "Processed real data")
            
        else:
            print(f"Real data file not found: {data_file}")
            return True  # Skip this test
            
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        return False
    
    return True

def main():
    """Run all debugging tests"""
    print("Starting systematic NaN debugging...")
    
    tests = [
        ("Preprocessing Pipeline", test_preprocessing_pipeline),
        ("Data Generation", test_data_generation), 
        ("Model Initialization", test_model_initialization),
        ("Forward Pass", test_forward_pass),
        ("Real Data Pipeline", test_real_data_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("DEBUGGING SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The NaN issue may be in training dynamics.")
        print("Next steps: Check batch data preparation and training loop.")
    else:
        print("\n🔍 Found issues! Focus on fixing the failed components first.")
    
    return all_passed

if __name__ == "__main__":
    main()
