#!/usr/bin/env python3
"""
Updated diagnostic script for current training state
Tests the exact current codebase to isolate remaining NaN sources
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
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_stats(data, name):
    """Enhanced data checking with more detailed statistics"""
    has_nan = np.any(np.isnan(data))
    has_inf = np.any(np.isinf(data))
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    print(f"\n{name} detailed statistics:")
    print(f"  Shape: {data.shape}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Range: [{min_val:.6e}, {max_val:.6e}]")
    print(f"  Mean: {mean_val:.6e}")
    print(f"  Std: {std_val:.6e}")
    
    # Check for extreme values that could cause numerical issues
    if np.any(data > 1e6):
        print(f"  ⚠️  WARNING: {name} contains very large values (>1e6)")
    if np.any(data < -1e6):
        print(f"  ⚠️  WARNING: {name} contains very small values (<-1e6)")
    if std_val > 1e3:
        print(f"  ⚠️  WARNING: {name} has very high variance")
    
    if has_nan or has_inf:
        print(f"  ❌ PROBLEM: {name} contains NaN or Inf!")
        return False
    else:
        print(f"  ✅ {name} is numerically clean")
        return True

def test_current_data_generation():
    """Test current data generation with setigen fixes"""
    print("="*60)
    print("TESTING CURRENT DATA GENERATION")
    print("="*60)
    
    config = Config()
    
    # Create small background dataset exactly as in training
    print("Creating small background dataset...")
    background_data = np.random.randn(50, 6, 16, 512).astype(np.float32) + 1000.0
    
    # Apply the same preprocessing as in training
    print("Applying background preprocessing...")
    for i in range(background_data.shape[0]):
        for j in range(6):
            background_data[i, j] = pre_proc(background_data[i, j])
    
    check_data_stats(background_data, "Preprocessed background")
    
    # Test data generator with current implementation
    print("\n--- Testing current DataGenerator ---")
    try:
        generator = DataGenerator(config, background_data)
        
        # Generate small batch
        print("Generating training batch...")
        batch_data = generator.generate_training_batch(12)  # Small batch
        
        # Check each component with detailed stats
        all_clean = True
        for key, data in batch_data.items():
            clean = check_data_stats(data, f"Generated {key}")
            if not clean:
                all_clean = False
        
        return all_clean
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_current_model_components():
    """Test each loss component individually with current model"""
    print("\n" + "="*60)
    print("TESTING CURRENT MODEL COMPONENTS")
    print("="*60)
    
    config = Config()
    
    # Create model with current settings
    vae = create_vae_model(config)
    
    print(f"Model created with alpha={vae.alpha}, beta={vae.beta}")
    
    # Create test data in correct format
    batch_size = 8
    test_data = np.random.randn(batch_size, 6, 16, 512).astype(np.float32) * 0.1 + 0.5
    
    # Normalize test data
    for i in range(batch_size):
        for j in range(6):
            test_data[i, j] = pre_proc(test_data[i, j])
    
    check_data_stats(test_data, "Test input data")
    
    print("\n--- Testing Encoder ---")
    try:
        # Test encoder with proper reshaping
        encoder_input = test_data.reshape(batch_size * 6, 16, 512, 1)
        z_mean, z_log_var, z = vae.encoder(encoder_input, training=False)
        
        check_data_stats(z_mean.numpy(), "Encoder z_mean")
        check_data_stats(z_log_var.numpy(), "Encoder z_log_var")
        check_data_stats(z.numpy(), "Encoder z")
        
        # Check for extreme values in latent space
        if np.any(np.abs(z_log_var.numpy()) > 10):
            print("  ⚠️  WARNING: z_log_var contains extreme values (>10)")
        
    except Exception as e:
        print(f"❌ Encoder test failed: {e}")
        return False
    
    print("\n--- Testing Decoder ---")
    try:
        reconstruction = vae.decoder(z, training=False)
        check_data_stats(reconstruction.numpy(), "Decoder output")
        
    except Exception as e:
        print(f"❌ Decoder test failed: {e}")
        return False
    
    print("\n--- Testing Individual Loss Components ---")
    
    # Test KL Loss (this is showing inf in logs)
    print("\nTesting KL Loss:")
    try:
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_per_sample = tf.reduce_sum(kl_loss, axis=1)
        kl_total = tf.reduce_mean(kl_per_sample)
        
        print(f"  z_log_var range: [{np.min(z_log_var.numpy()):.6f}, {np.max(z_log_var.numpy()):.6f}]")
        print(f"  exp(z_log_var) range: [{np.min(tf.exp(z_log_var).numpy()):.6e}, {np.max(tf.exp(z_log_var).numpy()):.6e}]")
        print(f"  KL per sample: {kl_per_sample.numpy()}")
        print(f"  KL total: {kl_total.numpy():.6f}")
        
        if np.isnan(kl_total.numpy()) or np.isinf(kl_total.numpy()):
            print("  ❌ KL loss is NaN/Inf")
            return False
        else:
            print("  ✅ KL loss is finite")
            
    except Exception as e:
        print(f"❌ KL loss test failed: {e}")
        return False
    
    # Test Reconstruction Loss
    print("\nTesting Reconstruction Loss:")
    try:
        reconstruction_reshaped = tf.reshape(reconstruction, (batch_size, 6, 16, 512))
        
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(test_data, reconstruction_reshaped), 
                axis=(1, 2)
            )
        )
        
        print(f"  Reconstruction loss: {recon_loss.numpy():.6f}")
        print(f"  Target range: [{np.min(test_data):.6f}, {np.max(test_data):.6f}]")
        print(f"  Output range: [{np.min(reconstruction_reshaped.numpy()):.6f}, {np.max(reconstruction_reshaped.numpy()):.6f}]")
        
        if np.isnan(recon_loss.numpy()) or np.isinf(recon_loss.numpy()):
            print("  ❌ Reconstruction loss is NaN/Inf")
            return False
        else:
            print("  ✅ Reconstruction loss is finite")
            
    except Exception as e:
        print(f"❌ Reconstruction loss test failed: {e}")
        return False
    
    # Test Clustering Losses
    print("\nTesting Clustering Losses:")
    try:
        true_loss = vae.compute_clustering_loss_true(test_data)
        false_loss = vae.compute_clustering_loss_false(test_data)
        
        print(f"  True clustering loss: {true_loss.numpy():.6f}")
        print(f"  False clustering loss: {false_loss.numpy():.6f}")
        
        if np.isnan(true_loss.numpy()) or np.isinf(true_loss.numpy()):
            print("  ❌ True clustering loss is NaN/Inf")
            return False
        if np.isnan(false_loss.numpy()) or np.isinf(false_loss.numpy()):
            print("  ❌ False clustering loss is NaN/Inf")
            return False
        
        print("  ✅ Clustering losses are finite")
        
    except Exception as e:
        print(f"❌ Clustering loss test failed: {e}")
        return False
    
    # Test Combined Loss (as in training)
    print("\nTesting Combined Loss:")
    try:
        total_loss = (recon_loss + 
                     vae.beta * kl_total + 
                     vae.alpha * (true_loss + false_loss))
        
        print(f"  Components:")
        print(f"    Reconstruction: {recon_loss.numpy():.6f}")
        print(f"    KL (β={vae.beta}): {(vae.beta * kl_total).numpy():.6f}")
        print(f"    Clustering (α={vae.alpha}): {(vae.alpha * (true_loss + false_loss)).numpy():.6f}")
        print(f"  Total loss: {total_loss.numpy():.6f}")
        
        if np.isnan(total_loss.numpy()) or np.isinf(total_loss.numpy()):
            print("  ❌ Total loss is NaN/Inf")
            return False
        else:
            print("  ✅ Total loss is finite")
            
    except Exception as e:
        print(f"❌ Combined loss test failed: {e}")
        return False
    
    return True

def test_distributed_training_step():
    """Test distributed training step with current setup"""
    print("\n" + "="*60)
    print("TESTING DISTRIBUTED TRAINING STEP")
    print("="*60)
    
    # Skip if only one GPU available
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) < 2:
        print("Single GPU detected, testing regular training step...")
        strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Multi-GPU detected, testing with {strategy.num_replicas_in_sync} replicas...")
    
    config = Config()
    
    with strategy.scope():
        # Create model within strategy scope
        vae = create_vae_model(config)
        
        # Create small training data exactly as in pipeline
        background_data = np.random.randn(100, 6, 16, 512).astype(np.float32) + 1000.0
        
        # Apply preprocessing
        for i in range(background_data.shape[0]):
            for j in range(6):
                background_data[i, j] = pre_proc(background_data[i, j])
        
        generator = DataGenerator(config, background_data)
        train_data = generator.generate_training_batch(64)  # Small batch
        
        # Prepare data exactly as in training.py
        x_train = (train_data['concatenated'], train_data['true'], train_data['false'])
        y_train = train_data['concatenated']
        
        # Create distributed dataset
        batch_size = 32  # Smaller for testing
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.batch(batch_size)
        dataset = strategy.experimental_distribute_dataset(dataset)
        
        print("Testing single distributed training step...")
        
        @tf.function
        def distributed_test_step(dist_inputs):
            def step_fn(inputs):
                x, y = inputs
                
                # Forward pass exactly as in VAE train_step
                batch_size_local = tf.shape(x[0])[0]
                encoder_input = tf.reshape(x[0], (batch_size_local * 6, 16, 512, 1))
                z_mean, z_log_var, z = vae.encoder(encoder_input, training=True)
                reconstruction = vae.decoder(z, training=True)
                reconstruction = tf.reshape(reconstruction, tf.shape(y))
                
                # Individual loss components
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
                    )
                )
                
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                
                false_loss = vae.compute_clustering_loss_false(x[2])
                true_loss = vae.compute_clustering_loss_true(x[1])
                
                total_loss = (reconstruction_loss + 
                             vae.beta * kl_loss + 
                             vae.alpha * (true_loss + false_loss))
                
                return {
                    'total_loss': total_loss,
                    'reconstruction_loss': reconstruction_loss,
                    'kl_loss': kl_loss,
                    'true_loss': true_loss,
                    'false_loss': false_loss,
                    'z_mean_sample': tf.reduce_mean(z_mean),
                    'z_log_var_sample': tf.reduce_mean(z_log_var)
                }
            
            return strategy.run(step_fn, args=(dist_inputs,))
        
        # Test with first batch
        for dist_inputs in dataset:
            try:
                results = distributed_test_step(dist_inputs)
                
                print("Distributed step results:")
                for key in results:
                    if strategy.num_replicas_in_sync > 1:
                        values = strategy.experimental_local_results(results[key])
                        print(f"  {key}:")
                        for i, val in enumerate(values):
                            val_scalar = float(val.numpy())
                            is_bad = np.isnan(val_scalar) or np.isinf(val_scalar)
                            print(f"    Replica {i}: {val_scalar:.6e} {'[NaN/Inf!]' if is_bad else ''}")
                            if is_bad:
                                return False
                    else:
                        val_scalar = float(results[key].numpy())
                        is_bad = np.isnan(val_scalar) or np.isinf(val_scalar)
                        print(f"  {key}: {val_scalar:.6e} {'[NaN/Inf!]' if is_bad else ''}")
                        if is_bad:
                            return False
                
                print("✅ Distributed training step completed successfully!")
                break  # Only test first batch
                
            except Exception as e:
                print(f"❌ Distributed step failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    return True

def test_problematic_scenarios():
    """Test scenarios that commonly cause NaN"""
    print("\n" + "="*60)
    print("TESTING PROBLEMATIC SCENARIOS")
    print("="*60)
    
    config = Config()
    vae = create_vae_model(config)
    
    scenarios = [
        ("Very small values", np.random.randn(4, 6, 16, 512) * 1e-10 + 1e-9),
        ("Very large values", np.random.randn(4, 6, 16, 512) * 1000 + 10000),
        ("Mixed extreme values", np.concatenate([
            np.random.randn(2, 6, 16, 512) * 1e-10 + 1e-9,
            np.random.randn(2, 6, 16, 512) * 1000 + 10000
        ])),
        ("Zero values", np.zeros((4, 6, 16, 512))),
        ("All ones", np.ones((4, 6, 16, 512)))
    ]
    
    for name, test_data in scenarios:
        print(f"\n--- Testing {name} ---")
        
        # Apply preprocessing
        processed_data = np.zeros_like(test_data)
        for i in range(test_data.shape[0]):
            for j in range(6):
                processed_data[i, j] = pre_proc(test_data[i, j])
        
        check_data_stats(processed_data, f"{name} after preprocessing")
        
        # Test forward pass
        try:
            encoder_input = processed_data.reshape(-1, 16, 512, 1)
            z_mean, z_log_var, z = vae.encoder(encoder_input, training=False)
            
            # Check latent space
            if np.any(np.abs(z_log_var.numpy()) > 20):
                print(f"  ⚠️  {name}: Extreme z_log_var values detected")
            
            # Test KL loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            if np.isnan(kl_loss.numpy()) or np.isinf(kl_loss.numpy()):
                print(f"  ❌ {name}: KL loss is NaN/Inf")
            else:
                print(f"  ✅ {name}: KL loss is finite ({kl_loss.numpy():.6f})")
                
        except Exception as e:
            print(f"  ❌ {name}: Failed with error: {e}")

def main():
    """Run comprehensive diagnostic"""
    print("Starting comprehensive training diagnostic...")
    
    tests = [
        ("Current Data Generation", test_current_data_generation),
        ("Current Model Components", test_current_model_components),
        ("Distributed Training Step", test_distributed_training_step),
        ("Problematic Scenarios", test_problematic_scenarios)
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
    print("COMPREHENSIVE DIAGNOSTIC SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The remaining NaN issue may be in specific edge cases.")
        print("Recommendations:")
        print("1. Consider reducing batch size to avoid memory pressure")
        print("2. Add gradient clipping to prevent gradient explosion")
        print("3. Use mixed precision training for numerical stability")
    else:
        print("\n🔍 Found issues! Focus on the failed components.")
        print("Key areas to investigate:")
        print("1. KL loss computation (check for extreme z_log_var values)")
        print("2. Reconstruction loss (check input/output ranges)")
        print("3. Distributed training synchronization")
    
    return all_passed

if __name__ == "__main__":
    main()
