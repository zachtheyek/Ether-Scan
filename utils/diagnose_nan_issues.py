"""
Diagnostic script to isolate and test NaN issues in the SETI training pipeline
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
import logging
from datetime import datetime

from config import Config
from preprocessing import DataPreprocessor, normalize_log
from data_generation import DataGenerator, create_mixed_training_batch
from models.vae import create_vae_model, BetaVAE
from training import TrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_normalize_log():
    """Test the normalize_log function for numerical stability"""
    logger.info("Testing normalize_log function...")
    
    # Test cases that could cause issues
    test_cases = [
        ("Normal data", np.random.randn(16, 512) * 100 + 1000),
        ("Very small values", np.random.randn(16, 512) * 1e-10 + 1e-9),
        ("Very large values", np.random.randn(16, 512) * 1e10 + 1e11),
        ("Negative values", np.random.randn(16, 512) * 100 - 50),
        ("Zero values", np.zeros((16, 512))),
        ("Mixed extreme", np.concatenate([
            np.full((8, 512), 1e-12),
            np.full((8, 512), 1e12)
        ]))
    ]
    
    for name, data in test_cases:
        try:
            normalized = normalize_log(data)
            has_nan = np.any(np.isnan(normalized))
            has_inf = np.any(np.isinf(normalized))
            min_val, max_val = np.min(normalized), np.max(normalized)
            
            logger.info(f"{name}: NaN={has_nan}, Inf={has_inf}, range=[{min_val:.4f}, {max_val:.4f}]")
            
            if has_nan or has_inf:
                logger.error(f"FAILED: {name} produced NaN/Inf values")
                return False
                
        except Exception as e:
            logger.error(f"FAILED: {name} raised exception: {e}")
            return False
    
    logger.info("normalize_log test PASSED")
    return True

def test_kl_loss_computation():
    """Test KL divergence computation for numerical stability"""
    logger.info("Testing KL divergence computation...")
    
    # Test various latent vector scenarios
    batch_size = 128
    latent_dim = 8
    
    test_cases = [
        ("Normal distribution", tf.random.normal((batch_size, latent_dim)), 
         tf.random.normal((batch_size, latent_dim))),
        ("Large mean values", tf.random.normal((batch_size, latent_dim)) * 10, 
         tf.random.normal((batch_size, latent_dim))),
        ("Large variance values", tf.random.normal((batch_size, latent_dim)), 
         tf.random.normal((batch_size, latent_dim)) * 5),
        ("Very small variance", tf.random.normal((batch_size, latent_dim)), 
         tf.random.normal((batch_size, latent_dim)) * 0.01),
        ("Extreme values", tf.random.normal((batch_size, latent_dim)) * 100, 
         tf.random.normal((batch_size, latent_dim)) * 10)
    ]
    
    for name, z_mean, z_log_var in test_cases:
        try:
            # Current implementation (problematic)
            z_mean_clamped = tf.clip_by_value(z_mean, -5.0, 5.0)
            z_log_var_clamped = tf.clip_by_value(z_log_var, -10.0, 2.0)
            kl_loss_current = -0.5 * (1 + z_log_var_clamped - tf.square(z_mean_clamped) - tf.exp(z_log_var_clamped))
            kl_loss_current = tf.reduce_mean(tf.reduce_sum(kl_loss_current, axis=1))
            
            # Improved implementation (more stable)
            z_mean_stable = tf.clip_by_value(z_mean, -3.0, 3.0)
            z_log_var_stable = tf.clip_by_value(z_log_var, -6.0, 1.0)  # Tighter bounds
            kl_loss_stable = -0.5 * (1 + z_log_var_stable - tf.square(z_mean_stable) - tf.exp(z_log_var_stable))
            kl_loss_stable = tf.reduce_mean(tf.reduce_sum(kl_loss_stable, axis=1))
            kl_loss_stable = tf.clip_by_value(kl_loss_stable, 0.0, 10.0)  # Final clipping
            
            current_val = float(kl_loss_current.numpy())
            stable_val = float(kl_loss_stable.numpy())
            
            has_nan_current = np.isnan(current_val) or np.isinf(current_val)
            has_nan_stable = np.isnan(stable_val) or np.isinf(stable_val)
            
            logger.info(f"{name}: Current={current_val:.4f} (NaN/Inf: {has_nan_current}), "
                       f"Stable={stable_val:.4f} (NaN/Inf: {has_nan_stable})")
            
            if has_nan_current and not has_nan_stable:
                logger.info(f"  -> Stability improvement achieved for {name}")
            elif has_nan_stable:
                logger.warning(f"  -> Still unstable: {name}")
                
        except Exception as e:
            logger.error(f"FAILED: {name} raised exception: {e}")
            return False
    
    logger.info("KL loss test completed")
    return True

def test_clustering_loss():
    """Test clustering loss computation for numerical stability"""
    logger.info("Testing clustering loss computation...")
    
    batch_size = 32
    latent_dim = 8
    
    # Create test latent vectors for 6 observations
    latent_vectors = [tf.random.normal((batch_size, latent_dim)) for _ in range(6)]
    
    # Test current dissimilarity loss
    def compute_similarity_loss_current(a, b):
        diff = a - b
        diff = tf.clip_by_value(diff, -10.0, 10.0)
        distance = tf.norm(diff, axis=1)
        distance = tf.clip_by_value(distance, 1e-8, 10.0)
        return tf.reduce_mean(distance)
    
    def compute_dissimilarity_loss_current(a, b):
        similarity = compute_similarity_loss_current(a, b)
        return -tf.math.log(similarity + 1e-8)
    
    # Test improved dissimilarity loss
    def compute_dissimilarity_loss_stable(a, b):
        diff = a - b
        diff = tf.clip_by_value(diff, -5.0, 5.0)
        distance = tf.norm(diff, axis=1)
        distance = tf.clip_by_value(distance, 1e-6, 5.0)  # Tighter bounds
        mean_distance = tf.reduce_mean(distance)
        # Use negative log with better numerical stability
        return -tf.math.log(tf.maximum(mean_distance, 1e-6))
    
    try:
        # Test with various scenarios
        test_vectors = [
            ("Normal", [tf.random.normal((batch_size, latent_dim)) for _ in range(6)]),
            ("Very similar", [tf.random.normal((batch_size, latent_dim)) * 0.01 for _ in range(6)]),
            ("Very different", [tf.random.normal((batch_size, latent_dim)) * 10 for _ in range(6)]),
        ]
        
        for name, vectors in test_vectors:
            # Test current vs stable implementation
            current_loss = compute_dissimilarity_loss_current(vectors[0], vectors[1])
            stable_loss = compute_dissimilarity_loss_stable(vectors[0], vectors[1])
            
            current_val = float(current_loss.numpy())
            stable_val = float(stable_loss.numpy())
            
            has_nan_current = np.isnan(current_val) or np.isinf(current_val)
            has_nan_stable = np.isnan(stable_val) or np.isinf(stable_val)
            
            logger.info(f"Dissimilarity {name}: Current={current_val:.4f} (NaN/Inf: {has_nan_current}), "
                       f"Stable={stable_val:.4f} (NaN/Inf: {has_nan_stable})")
        
    except Exception as e:
        logger.error(f"Clustering loss test failed: {e}")
        return False
    
    logger.info("Clustering loss test completed")
    return True

def test_data_generation():
    """Test data generation pipeline for numerical issues"""
    logger.info("Testing data generation pipeline...")
    
    try:
        # Create minimal config and background data
        config = Config()
        
        # Create dummy background data
        background_data = np.random.randn(100, 6, 16, 512).astype(np.float32)
        
        # Apply log normalization
        for i in range(background_data.shape[0]):
            for j in range(6):
                background_data[i, j] = normalize_log(background_data[i, j])
        
        # Test data generator
        generator = DataGenerator(config, background_data)
        
        # Generate small batch
        test_batch = create_mixed_training_batch(generator, 8)
        concatenated, true_data, false_data = test_batch
        
        # Check for NaN/Inf in generated data
        for name, data in [("concatenated", concatenated), ("true", true_data), ("false", false_data)]:
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            min_val, max_val = np.min(data), np.max(data)
            
            logger.info(f"Generated {name}: NaN={has_nan}, Inf={has_inf}, range=[{min_val:.4f}, {max_val:.4f}]")
            
            if has_nan or has_inf:
                logger.error(f"FAILED: Generated data contains NaN/Inf")
                return False
        
    except Exception as e:
        logger.error(f"Data generation test failed: {e}")
        return False
    
    logger.info("Data generation test PASSED")
    return True

def test_single_training_step():
    """Test a single training step to isolate where NaN occurs"""
    logger.info("Testing single training step...")
    
    try:
        config = Config()
        
        # Create minimal background data
        background_data = np.random.randn(50, 6, 16, 512).astype(np.float32)
        
        # Apply normalization
        for i in range(background_data.shape[0]):
            for j in range(6):
                background_data[i, j] = normalize_log(background_data[i, j])
        
        # Create model
        with tf.distribute.get_strategy().scope():
            vae = create_vae_model(config)
            
            # Use more conservative optimizer settings
            vae.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.0001,  # Much smaller learning rate
                    clipnorm=0.5,  # Tighter gradient clipping
                    epsilon=1e-8
                )
            )
        
        # Generate training batch
        generator = DataGenerator(config, background_data)
        batch_data = create_mixed_training_batch(generator, 16)  # Small batch
        
        # Format data properly
        concatenated, true_data, false_data = batch_data
        inputs = (concatenated, true_data, false_data)
        target = concatenated
        
        logger.info("Input data shapes:")
        logger.info(f"  Concatenated: {concatenated.shape}")
        logger.info(f"  True: {true_data.shape}")
        logger.info(f"  False: {false_data.shape}")
        logger.info(f"  Target: {target.shape}")
        
        # Check input data for issues
        for name, data in [("concatenated", concatenated), ("true", true_data), ("false", false_data)]:
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            if has_nan or has_inf:
                logger.error(f"Input data {name} contains NaN/Inf")
                return False
        
        # Perform training step
        logger.info("Performing single training step...")
        
        # Monitor loss components
        for step in range(5):  # Test multiple steps
            result = vae.train_step((inputs, target))
            
            logger.info(f"Step {step + 1}:")
            for key, value in result.items():
                val = float(value.numpy())
                is_bad = np.isnan(val) or np.isinf(val)
                logger.info(f"  {key}: {val:.6f} {'[NaN/Inf!]' if is_bad else ''}")
                
                if is_bad:
                    logger.error(f"NaN/Inf detected in {key} at step {step + 1}")
                    return False
            
            logger.info("")
        
    except Exception as e:
        logger.error(f"Single training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("Single training step test PASSED")
    return True

def main():
    """Run all diagnostic tests"""
    logger.info("="*60)
    logger.info("Starting NaN Diagnostic Tests")
    logger.info("="*60)
    
    tests = [
        ("Normalize Log Function", test_normalize_log),
        ("KL Loss Computation", test_kl_loss_computation),
        ("Clustering Loss", test_clustering_loss),
        ("Data Generation", test_data_generation),
        ("Single Training Step", test_single_training_step)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    if not all_passed:
        logger.info("\nRecommendations:")
        logger.info("1. Apply fixes to models/vae.py for KL loss stability")
        logger.info("2. Improve clustering loss numerical stability") 
        logger.info("3. Use more conservative training parameters")
        logger.info("4. Implement better gradient clipping")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)