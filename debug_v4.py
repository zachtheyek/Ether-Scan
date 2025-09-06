#!/usr/bin/env python3
"""
Quick verification script to test the NaN fixes
Run this after applying the fixes to verify they work
"""

import sys
import os
sys.path.append('.')

import numpy as np
import tensorflow as tf
from config import Config
from data_generation import DataGenerator
from models.vae import create_vae_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_background_loading():
    """Test the fixed background loading"""
    logger.info("Testing fixed background data loading...")
    
    # Import the fixed function
    from main import load_background_data
    
    config = Config()
    
    try:
        # Load small amount of background data
        background_data = load_background_data(config)
        
        # Check value ranges
        min_val = np.min(background_data)
        max_val = np.max(background_data)
        mean_val = np.mean(background_data)
        
        logger.info(f"Background data range: [{min_val:.6f}, {max_val:.6f}]")
        logger.info(f"Background mean: {mean_val:.6f}")
        
        if max_val <= 2.0:
            logger.info("✅ Background data properly normalized")
            return True
        else:
            logger.error(f"❌ Background data still too large: {max_val}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Background loading failed: {e}")
        return False

def verify_data_generation():
    """Test data generation with normalized backgrounds"""
    logger.info("Testing data generation with normalized backgrounds...")
    
    config = Config()
    
    # Create small normalized background for testing
    background_data = np.random.rand(10, 6, 16, 512).astype(np.float32)
    
    try:
        generator = DataGenerator(config, background_data)
        batch_data = generator.generate_training_batch(4)
        
        # Check all components
        for key, data in batch_data.items():
            min_val = np.min(data)
            max_val = np.max(data)
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            
            logger.info(f"{key}: range=[{min_val:.6f}, {max_val:.6f}], NaN={has_nan}, Inf={has_inf}")
            
            if has_nan or has_inf or max_val > 10:
                logger.error(f"❌ {key} has bad values!")
                return False
        
        logger.info("✅ Data generation produces clean values")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data generation failed: {e}")
        return False

def verify_model_forward_pass():
    """Test model forward pass with clean data"""
    logger.info("Testing model forward pass...")
    
    config = Config()
    
    # Create model
    vae = create_vae_model(config)
    
    # Create clean test data
    test_data = np.random.rand(2, 6, 16, 512).astype(np.float32)
    
    try:
        # Test encoding
        batch_size = tf.shape(test_data)[0]
        encoder_input = tf.reshape(test_data, (batch_size * 6, 16, 512, 1))
        z_mean, z_log_var, z = vae.encoder(encoder_input, training=False)
        
        # Check encoder outputs
        for name, tensor in [("z_mean", z_mean), ("z_log_var", z_log_var), ("z", z)]:
            vals = tensor.numpy()
            has_nan = np.any(np.isnan(vals))
            has_inf = np.any(np.isinf(vals))
            min_val = np.min(vals)
            max_val = np.max(vals)
            
            logger.info(f"{name}: range=[{min_val:.6f}, {max_val:.6f}], NaN={has_nan}, Inf={has_inf}")
            
            if has_nan or has_inf:
                logger.error(f"❌ {name} has NaN/Inf!")
                return False
        
        # Test reconstruction
        reconstruction = vae.decoder(z, training=False)
        reconstruction = tf.reshape(reconstruction, (2, 6, 16, 512))
        
        recon_vals = reconstruction.numpy()
        has_nan = np.any(np.isnan(recon_vals))
        has_inf = np.any(np.isinf(recon_vals))
        
        logger.info(f"Reconstruction: NaN={has_nan}, Inf={has_inf}")
        
        if has_nan or has_inf:
            logger.error(f"❌ Reconstruction has NaN/Inf!")
            return False
        
        logger.info("✅ Model forward pass produces clean values")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model forward pass failed: {e}")
        return False

def verify_loss_computation():
    """Test loss computation with clean data"""
    logger.info("Testing loss computation...")
    
    config = Config()
    vae = create_vae_model(config)
    
    # Clean test data
    test_data = np.random.rand(2, 6, 16, 512).astype(np.float32) * 0.5 + 0.25  # Keep in [0.25, 0.75]
    
    try:
        # Test clustering losses
        true_loss = vae.compute_clustering_loss_true(test_data)
        false_loss = vae.compute_clustering_loss_false(test_data)
        
        true_val = float(true_loss.numpy())
        false_val = float(false_loss.numpy())
        
        logger.info(f"True clustering loss: {true_val:.6f}")
        logger.info(f"False clustering loss: {false_val:.6f}")
        
        if np.isnan(true_val) or np.isinf(true_val):
            logger.error("❌ True clustering loss is NaN/Inf!")
            return False
        
        if np.isnan(false_val) or np.isinf(false_val):
            logger.error("❌ False clustering loss is NaN/Inf!")
            return False
        
        # Test full loss computation
        with tf.GradientTape() as tape:
            batch_size = tf.shape(test_data)[0]
            encoder_input = tf.reshape(test_data, (batch_size * 6, 16, 512, 1))
            z_mean, z_log_var, z = vae.encoder(encoder_input, training=True)
            reconstruction = vae.decoder(z, training=True)
            reconstruction = tf.reshape(reconstruction, tf.shape(test_data))
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(test_data, reconstruction), 
                    axis=(1, 2)
                )
            )
            
            # KL loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + vae.beta * kl_loss + vae.alpha * (true_loss + false_loss)
        
        # Check all losses
        losses = {
            'reconstruction': float(reconstruction_loss.numpy()),
            'kl': float(kl_loss.numpy()),
            'total': float(total_loss.numpy())
        }
        
        for name, value in losses.items():
            logger.info(f"{name} loss: {value:.6f}")
            if np.isnan(value) or np.isinf(value):
                logger.error(f"❌ {name} loss is NaN/Inf!")
                return False
        
        logger.info("✅ All losses are finite!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Loss computation failed: {e}")
        return False

def main():
    """Run all verification tests"""
    logger.info("="*60)
    logger.info("VERIFYING NaN FIXES")
    logger.info("="*60)
    
    tests = [
        ("Background Loading", verify_background_loading),
        ("Data Generation", verify_data_generation),
        ("Model Forward Pass", verify_model_forward_pass),
        ("Loss Computation", verify_loss_computation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All fixes verified! Training should work now.")
    else:
        logger.info("\n🔍 Some fixes need more work.")
    
    return all_passed

if __name__ == "__main__":
    main()
