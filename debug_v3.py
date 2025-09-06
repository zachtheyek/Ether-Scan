#!/usr/bin/env python3
"""
Updated diagnostic script for current training state
Tests the actual codebase with real data to isolate remaining NaN issues
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

def load_current_background_data(n_samples=1000):
    """Load background data exactly as current main.py does"""
    config = Config()
    
    # Exact loading as in main.py
    all_backgrounds = []
    target_backgrounds = n_samples
    chunk_size = 150
    downsample_factor = 8
    final_width = 512
    
    for filename in config.data.training_files[:1]:  # Just first file for testing
        filepath = config.get_training_file_path(filename)
        if not os.path.exists(filepath):
            continue
            
        logger.info(f"Loading {filename}")
        raw_data = np.load(filepath, mmap_mode='r')
        
        # Apply exact subset
        start, end = config.get_file_subset(filename)
        if start is not None or end is not None:
            raw_data = raw_data[start:end]
        
        # Process exactly as main.py
        n_chunks = min(10, (raw_data.shape[0] + chunk_size - 1) // chunk_size)
        
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, raw_data.shape[0])
            
            chunk_data = np.array(raw_data[chunk_start:chunk_end])
            
            for cadence_idx in range(chunk_data.shape[0]):
                if len(all_backgrounds) >= target_backgrounds:
                    break
                    
                cadence = chunk_data[cadence_idx]
                
                # Skip invalid cadences
                if np.any(np.isnan(cadence)) or np.any(np.isinf(cadence)) or np.max(cadence) <= 0:
                    continue
                
                # Exact downsampling as in main.py
                from skimage.transform import downscale_local_mean
                downsampled_cadence = np.zeros((6, 16, final_width), dtype=np.float32)
                for obs_idx in range(6):
                    downsampled_cadence[obs_idx] = downscale_local_mean(
                        cadence[obs_idx], (1, downsample_factor)
                    ).astype(np.float32)
                
                all_backgrounds.append(downsampled_cadence)
            
            del chunk_data
            gc.collect()
            
            if len(all_backgrounds) >= target_backgrounds:
                break
    
    background_array = np.array(all_backgrounds, dtype=np.float32)
    logger.info(f"Loaded background shape: {background_array.shape}")
    
    return background_array

def test_signal_injection_fix():
    """Test if our signal injection fix worked"""
    logger.info("="*60)
    logger.info("TESTING SIGNAL INJECTION FIX")
    logger.info("="*60)
    
    config = Config()
    background_data = load_current_background_data(100)
    
    logger.info("Testing data generation with current fixes...")
    generator = DataGenerator(config, background_data)
    
    # Generate small batch to test
    try:
        batch_data = generator.generate_training_batch(8)
        
        logger.info("Generated data analysis:")
        for key, data in batch_data.items():
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            min_val = np.min(data)
            max_val = np.max(data)
            mean_val = np.mean(data)
            
            logger.info(f"  {key}:")
            logger.info(f"    Shape: {data.shape}")
            logger.info(f"    NaN: {has_nan}, Inf: {has_inf}")
            logger.info(f"    Range: [{min_val:.6f}, {max_val:.6f}]")
            logger.info(f"    Mean: {mean_val:.6f}")
            
            if has_nan or has_inf:
                logger.error(f"    ❌ {key} still contains NaN/Inf!")
                return False
            elif max_val > 1000:  # Check if we still have massive values
                logger.error(f"    ❌ {key} still has massive values (expected ≤ ~2.0)!")
                return False
            else:
                logger.info(f"    ✅ {key} looks good")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Signal injection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_loss_components():
    """Test each loss component individually with current model"""
    logger.info("\n" + "="*60)
    logger.info("TESTING INDIVIDUAL LOSS COMPONENTS")
    logger.info("="*60)
    
    config = Config()
    
    # Create model exactly as in training
    with tf.distribute.MirroredStrategy().scope():
        vae = create_vae_model(config)
    
    # Generate test data with current pipeline
    background_data = load_current_background_data(50)
    generator = DataGenerator(config, background_data)
    
    try:
        batch_data = generator.generate_training_batch(4)
        
        concatenated = batch_data['concatenated']
        true_data = batch_data['true']
        false_data = batch_data['false']
        
        logger.info(f"Test data shapes:")
        logger.info(f"  Concatenated: {concatenated.shape}")
        logger.info(f"  True: {true_data.shape}")
        logger.info(f"  False: {false_data.shape}")
        
        # Test encoder forward pass
        logger.info("\n--- Testing Encoder ---")
        batch_size = concatenated.shape[0]
        encoder_input = tf.reshape(concatenated, (batch_size * 6, 16, 512, 1))
        
        z_mean, z_log_var, z = vae.encoder(encoder_input, training=False)
        
        logger.info(f"Encoder outputs:")
        for name, tensor in [("z_mean", z_mean), ("z_log_var", z_log_var), ("z", z)]:
            vals = tensor.numpy()
            has_nan = np.any(np.isnan(vals))
            has_inf = np.any(np.isinf(vals))
            logger.info(f"  {name}: NaN={has_nan}, Inf={has_inf}, range=[{np.min(vals):.6f}, {np.max(vals):.6f}]")
            if has_nan or has_inf:
                logger.error(f"❌ Encoder output {name} contains NaN/Inf!")
                return False
        
        # Test decoder
        logger.info("\n--- Testing Decoder ---")
        reconstruction = vae.decoder(z, training=False)
        reconstruction = tf.reshape(reconstruction, (batch_size, 6, 16, 512))
        
        recon_vals = reconstruction.numpy()
        has_nan = np.any(np.isnan(recon_vals))
        has_inf = np.any(np.isinf(recon_vals))
        logger.info(f"Decoder output: NaN={has_nan}, Inf={has_inf}, range=[{np.min(recon_vals):.6f}, {np.max(recon_vals):.6f}]")
        
        if has_nan or has_inf:
            logger.error(f"❌ Decoder output contains NaN/Inf!")
            return False
        
        # Test individual loss components
        logger.info("\n--- Testing Loss Components ---")
        
        # 1. Reconstruction Loss
        try:
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(concatenated, reconstruction), 
                    axis=(1, 2)
                )
            )
            recon_val = float(recon_loss.numpy())
            logger.info(f"1. Reconstruction loss: {recon_val:.6f}")
            if np.isnan(recon_val) or np.isinf(recon_val):
                logger.error("❌ Reconstruction loss is NaN/Inf!")
                return False
        except Exception as e:
            logger.error(f"❌ Reconstruction loss failed: {e}")
            return False
        
        # 2. KL Loss
        try:
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_val = float(kl_loss.numpy())
            logger.info(f"2. KL loss: {kl_val:.6f}")
            if np.isnan(kl_val) or np.isinf(kl_val):
                logger.error("❌ KL loss is NaN/Inf!")
                
                # Debug KL components
                logger.info("KL loss debugging:")
                z_mean_vals = z_mean.numpy()
                z_log_var_vals = z_log_var.numpy()
                logger.info(f"  z_mean range: [{np.min(z_mean_vals):.6f}, {np.max(z_mean_vals):.6f}]")
                logger.info(f"  z_log_var range: [{np.min(z_log_var_vals):.6f}, {np.max(z_log_var_vals):.6f}]")
                logger.info(f"  exp(z_log_var) range: [{np.min(np.exp(z_log_var_vals)):.6f}, {np.max(np.exp(z_log_var_vals)):.6f}]")
                return False
        except Exception as e:
            logger.error(f"❌ KL loss failed: {e}")
            return False
        
        # 3. Clustering Losses
        try:
            true_loss = vae.compute_clustering_loss_true(true_data)
            true_val = float(true_loss.numpy())
            logger.info(f"3. True clustering loss: {true_val:.6f}")
            if np.isnan(true_val) or np.isinf(true_val):
                logger.error("❌ True clustering loss is NaN/Inf!")
                return False
        except Exception as e:
            logger.error(f"❌ True clustering loss failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        try:
            false_loss = vae.compute_clustering_loss_false(false_data)
            false_val = float(false_loss.numpy())
            logger.info(f"4. False clustering loss: {false_val:.6f}")
            if np.isnan(false_val) or np.isinf(false_val):
                logger.error("❌ False clustering loss is NaN/Inf!")
                return False
        except Exception as e:
            logger.error(f"❌ False clustering loss failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. Total Loss
        try:
            total_loss = (recon_loss + 
                         vae.beta * kl_loss + 
                         vae.alpha * (true_loss + false_loss))
            total_val = float(total_loss.numpy())
            logger.info(f"5. Total loss: {total_val:.6f}")
            if np.isnan(total_val) or np.isinf(total_val):
                logger.error("❌ Total loss is NaN/Inf!")
                return False
            elif total_val > 10000:
                logger.warning(f"⚠️  Total loss is very large: {total_val:.6f}")
                logger.info(f"   Loss breakdown:")
                logger.info(f"   - Reconstruction: {recon_val:.6f}")
                logger.info(f"   - KL (β={vae.beta}): {vae.beta * kl_val:.6f}")
                logger.info(f"   - Clustering (α={vae.alpha}): {vae.alpha * (true_val + false_val):.6f}")
                
        except Exception as e:
            logger.error(f"❌ Total loss failed: {e}")
            return False
        
        logger.info("✅ All loss components are finite!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Loss component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_computation():
    """Test gradient computation without distributed training"""
    logger.info("\n" + "="*60)
    logger.info("TESTING GRADIENT COMPUTATION")
    logger.info("="*60)
    
    config = Config()
    
    # Test without distributed strategy first
    vae = create_vae_model(config)
    
    # Generate minimal test data
    background_data = load_current_background_data(20)
    generator = DataGenerator(config, background_data)
    
    try:
        batch_data = generator.generate_training_batch(2)  # Very small batch
        
        concatenated = batch_data['concatenated']
        true_data = batch_data['true']
        false_data = batch_data['false']
        
        logger.info("Testing single gradient step...")
        
        with tf.GradientTape() as tape:
            # Forward pass
            batch_size = tf.shape(concatenated)[0]
            encoder_input = tf.reshape(concatenated, (batch_size * 6, 16, 512, 1))
            z_mean, z_log_var, z = vae.encoder(encoder_input, training=True)
            reconstruction = vae.decoder(z, training=True)
            reconstruction = tf.reshape(reconstruction, tf.shape(concatenated))
            
            # Compute losses
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(concatenated, reconstruction), 
                    axis=(1, 2)
                )
            )
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            false_loss = vae.compute_clustering_loss_false(false_data)
            true_loss = vae.compute_clustering_loss_true(true_data)
            
            total_loss = (reconstruction_loss + 
                         vae.beta * kl_loss + 
                         vae.alpha * (true_loss + false_loss))
        
        # Check loss values before gradient computation
        logger.info(f"Pre-gradient loss values:")
        logger.info(f"  Reconstruction: {float(reconstruction_loss.numpy()):.6f}")
        logger.info(f"  KL: {float(kl_loss.numpy()):.6f}")
        logger.info(f"  True clustering: {float(true_loss.numpy()):.6f}")
        logger.info(f"  False clustering: {float(false_loss.numpy()):.6f}")
        logger.info(f"  Total: {float(total_loss.numpy()):.6f}")
        
        # Compute gradients
        grads = tape.gradient(total_loss, vae.trainable_weights)
        
        # Check gradients
        logger.info("Checking gradients...")
        nan_grad_count = 0
        inf_grad_count = 0
        total_grad_count = 0
        
        for i, grad in enumerate(grads):
            if grad is not None:
                total_grad_count += 1
                grad_vals = grad.numpy()
                has_nan = np.any(np.isnan(grad_vals))
                has_inf = np.any(np.isinf(grad_vals))
                
                if has_nan:
                    nan_grad_count += 1
                    logger.error(f"  Gradient {i}: Contains NaN!")
                if has_inf:
                    inf_grad_count += 1
                    logger.error(f"  Gradient {i}: Contains Inf!")
                    
                if i < 3:  # Log first few gradients
                    logger.info(f"  Gradient {i}: range=[{np.min(grad_vals):.6f}, {np.max(grad_vals):.6f}], NaN={has_nan}, Inf={has_inf}")
        
        logger.info(f"Gradient summary: {nan_grad_count} NaN, {inf_grad_count} Inf out of {total_grad_count} total")
        
        if nan_grad_count == 0 and inf_grad_count == 0:
            logger.info("✅ All gradients are finite!")
            return True
        else:
            logger.error(f"❌ Found {nan_grad_count + inf_grad_count} bad gradients!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distributed_training_step():
    """Test full distributed training step"""
    logger.info("\n" + "="*60)
    logger.info("TESTING DISTRIBUTED TRAINING STEP")
    logger.info("="*60)
    
    config = Config()
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        vae = create_vae_model(config)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.model.learning_rate))
    
    # Generate minimal data
    background_data = load_current_background_data(50)
    generator = DataGenerator(config, background_data)
    
    try:
        batch_data = generator.generate_training_batch(16)  # Small batch for distributed test
        
        concatenated = batch_data['concatenated']
        true_data = batch_data['true']
        false_data = batch_data['false']
        
        # Create dataset for distributed training
        x_train = (concatenated, true_data, false_data)
        y_train = concatenated
        
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.batch(8)  # Very small batch size
        dataset = strategy.experimental_distribute_dataset(dataset)
        
        # Test one training step
        logger.info("Testing distributed training step...")
        
        step_count = 0
        for dist_inputs in dataset:
            step_count += 1
            logger.info(f"Processing distributed batch {step_count}...")
            
            try:
                # Use train_on_batch to get loss values
                results = vae.train_on_batch(
                    strategy.experimental_local_results(dist_inputs)[0][0],  # Get first replica's input
                    strategy.experimental_local_results(dist_inputs)[0][1]   # Get first replica's target
                )
                
                logger.info(f"Distributed step {step_count} results:")
                for key, value in results.items():
                    val = float(value) if hasattr(value, 'numpy') else float(value)
                    is_bad = np.isnan(val) or np.isinf(val)
                    logger.info(f"  {key}: {val:.6f} {'[NaN/Inf!]' if is_bad else ''}")
                    
                    if is_bad:
                        logger.error(f"❌ NaN/Inf detected in {key}!")
                        return False
                
                logger.info("✅ Distributed training step completed successfully!")
                break  # Only test first batch
                
            except Exception as e:
                logger.error(f"❌ Distributed step {step_count} failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Distributed training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive diagnostic of current training state"""
    logger.info("="*60)
    logger.info("CURRENT TRAINING STATE DIAGNOSTIC")
    logger.info("="*60)
    
    tests = [
        ("Signal Injection Fix", test_signal_injection_fix),
        ("Individual Loss Components", test_individual_loss_components),
        ("Gradient Computation", test_gradient_computation),
        ("Distributed Training Step", test_distributed_training_step)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Training should be stable now.")
    else:
        logger.info("\n🔍 Found remaining issues. Focus on failed components.")
        
        # Provide specific recommendations based on failures
        if not results.get("Signal Injection Fix", True):
            logger.info("\n📋 Signal Injection still has issues:")
            logger.info("   - Check data value ranges")
            logger.info("   - Verify pre_proc normalization")
            logger.info("   - Test setigen integration")
        
        if not results.get("Individual Loss Components", True):
            logger.info("\n📋 Loss component issues:")
            logger.info("   - Check clustering loss implementation")
            logger.info("   - Verify KL loss bounds")
            logger.info("   - Test encoder/decoder output ranges")
        
        if not results.get("Gradient Computation", True):
            logger.info("\n📋 Gradient issues:")
            logger.info("   - Reduce loss weights further")
            logger.info("   - Add gradient clipping")
            logger.info("   - Check optimizer settings")
    
    return all_passed

if __name__ == "__main__":
    main()
