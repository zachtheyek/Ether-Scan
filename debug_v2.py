#!/usr/bin/env python3
"""
Debug the EXACT training conditions that cause NaN
Tests distributed training with real batch sizes and data
"""

import sys
import os
sys.path.append('.')

import numpy as np
import tensorflow as tf
from config import Config
from preprocessing import DataPreprocessor
from data_generation import DataGenerator
from models.vae import create_vae_model
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_exact_training_environment():
    """Setup identical to actual training"""
    # Exact distributed strategy setup
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"Distributed strategy: {strategy.num_replicas_in_sync} replicas")
    
    # Exact memory settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    return strategy

def load_exact_background_data():
    """Load background data exactly as in training"""
    config = Config()
    
    # Load EXACTLY as in main.py
    all_backgrounds = []
    target_backgrounds = 6000
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
        
        # Process in chunks exactly as in main.py
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

def test_exact_training_step():
    """Test one training step with exact conditions"""
    logger.info("Setting up exact training environment...")
    
    strategy = setup_exact_training_environment()
    config = Config()
    
    # Load exact background data
    background_data = load_exact_background_data()
    
    with strategy.scope():
        # Create model with exact settings
        vae = create_vae_model(config)
        
        # Exact optimizer settings from training.py
        vae.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.model.learning_rate  # 0.001
            )
        )
        
        logger.info("Generating training data with exact parameters...")
        
        # Generate data exactly as in training
        generator = DataGenerator(config, background_data)
        
        # Exact sample count and chunking
        n_samples = config.training.num_samples_train  # 5000
        train_data = generator.generate_training_batch(n_samples * 3)  # 15000
        
        # Exact train/validation split
        n_train = int(n_samples * 3 * 0.8)  # 12000
        
        train_concat = train_data['concatenated'][:n_train]
        train_true = train_data['true'][:n_train]
        train_false = train_data['false'][:n_train]
        
        logger.info(f"Training data shapes:")
        logger.info(f"  Concatenated: {train_concat.shape}")
        logger.info(f"  True: {train_true.shape}")
        logger.info(f"  False: {train_false.shape}")
        
        # Check data before training
        logger.info("Checking data before training step...")
        for name, data in [("concatenated", train_concat), ("true", train_true), ("false", train_false)]:
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            logger.info(f"  {name}: NaN={has_nan}, Inf={has_inf}, range=[{np.min(data):.6f}, {np.max(data):.6f}]")
        
        # Prepare data exactly as in training.py
        x_train = (train_concat, train_true, train_false)
        y_train = train_concat
        
        # Create dataset with exact batch size
        batch_size = config.training.batch_size  # 256
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.batch(batch_size)
        dataset = strategy.experimental_distribute_dataset(dataset)
        
        logger.info(f"Created distributed dataset with batch_size={batch_size}")
        
        # Test single training step with exact conditions
        logger.info("Performing single distributed training step...")
        
        @tf.function
        def distributed_train_step(dist_inputs):
            def step_fn(inputs):
                with tf.GradientTape() as tape:
                    x, y = inputs
                    true_data = x[1]
                    false_data = x[2] 
                    x_main = x[0]
                    
                    # Forward pass
                    batch_size_local = tf.shape(x_main)[0]
                    encoder_input = tf.reshape(x_main, (batch_size_local * 6, 16, 512, 1))
                    z_mean, z_log_var, z = vae.encoder(encoder_input, training=True)
                    reconstruction = vae.decoder(z, training=True)
                    reconstruction = tf.reshape(reconstruction, tf.shape(y))
                    
                    # Loss computation
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
                        )
                    )
                    
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                    
                    false_loss = vae.compute_clustering_loss_false(false_data)
                    true_loss = vae.compute_clustering_loss_true(true_data)
                    
                    total_loss = (reconstruction_loss + 
                                 vae.beta * kl_loss + 
                                 vae.alpha * (true_loss + false_loss))
                    
                    return {
                        'total_loss': total_loss,
                        'reconstruction_loss': reconstruction_loss,
                        'kl_loss': kl_loss,
                        'true_loss': true_loss,
                        'false_loss': false_loss
                    }
            
            return strategy.run(step_fn, args=(dist_inputs,))
        
        # Test with first batch
        step_count = 0
        for dist_inputs in dataset:
            step_count += 1
            logger.info(f"Testing distributed step {step_count}...")
            
            try:
                results = distributed_train_step(dist_inputs)
                
                # Check results on each replica
                for key in results:
                    values = strategy.experimental_local_results(results[key])
                    logger.info(f"  {key}:")
                    for i, val in enumerate(values):
                        val_scalar = float(val.numpy())
                        is_bad = np.isnan(val_scalar) or np.isinf(val_scalar)
                        logger.info(f"    Replica {i}: {val_scalar:.6f} {'[NaN/Inf!]' if is_bad else ''}")
                        
                        if is_bad:
                            logger.error(f"❌ NaN/Inf detected in {key} on replica {i}!")
                            return False
                
                # Test gradient computation
                logger.info("Testing gradient computation...")
                
                @tf.function
                def test_gradients(dist_inputs):
                    def grad_fn(inputs):
                        with tf.GradientTape() as tape:
                            x, y = inputs
                            # Simplified loss for gradient test
                            loss = tf.reduce_mean(tf.square(y - x[0]))
                        grads = tape.gradient(loss, vae.trainable_weights)
                        return loss, grads
                    return strategy.run(grad_fn, args=(dist_inputs,))
                
                loss, grads = test_gradients(dist_inputs)
                
                # Check gradients
                grad_list = strategy.experimental_local_results(grads)
                for i, replica_grads in enumerate(grad_list):
                    logger.info(f"  Replica {i} gradients:")
                    for j, grad in enumerate(replica_grads[:3]):  # Check first 3 layers
                        if grad is not None:
                            grad_vals = grad.numpy()
                            has_nan = np.any(np.isnan(grad_vals))
                            has_inf = np.any(np.isinf(grad_vals))
                            logger.info(f"    Layer {j}: NaN={has_nan}, Inf={has_inf}")
                            if has_nan or has_inf:
                                logger.error(f"❌ Bad gradients in layer {j} on replica {i}!")
                                return False
                
                logger.info("✅ Single distributed step completed successfully!")
                break  # Only test first batch
                
            except Exception as e:
                logger.error(f"❌ Distributed step failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return True

def test_loss_scaling():
    """Test if loss scaling is the issue"""
    logger.info("\n" + "="*60)
    logger.info("TESTING LOSS SCALING HYPOTHESIS")
    logger.info("="*60)
    
    config = Config()
    
    # Test with reduced loss weights
    logger.info("Testing with reduced alpha/beta...")
    
    original_alpha = config.model.alpha
    original_beta = config.model.beta
    
    # Drastically reduce loss weights
    config.model.alpha = 0.1  # Reduce from 10.0
    config.model.beta = 0.1   # Reduce from 1.5
    
    logger.info(f"Original: alpha={original_alpha}, beta={original_beta}")
    logger.info(f"Reduced:  alpha={config.model.alpha}, beta={config.model.beta}")
    
    # Test if this prevents NaN
    try:
        return test_exact_training_step()
    except Exception as e:
        logger.error(f"Reduced loss weights still failed: {e}")
        return False

def main():
    """Run exact training conditions debug"""
    logger.info("="*60)
    logger.info("DEBUGGING EXACT TRAINING CONDITIONS")
    logger.info("="*60)
    
    tests = [
        ("Exact Training Step", test_exact_training_step),
        ("Loss Scaling Test", test_loss_scaling)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
            
            if not success:
                logger.info("Found the issue! Focus on this component.")
                break
                
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    logger.info("\n" + "="*60)
    logger.info("DEBUGGING COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()
