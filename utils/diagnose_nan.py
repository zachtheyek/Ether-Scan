#!/usr/bin/env python3
"""
Systematic NaN Diagnostic Script for SETI ML Pipeline
This script traces the data flow and identifies exactly where NaN values originate.
"""

import numpy as np
import tensorflow as tf
import logging
import traceback
import sys
from typing import Dict, Any, Tuple
import warnings

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from config import Config
from preprocessing import DataPreprocessor, normalize_log
from data_generation import DataGenerator
from training import TrainingPipeline
from models.vae import create_vae_model

# Setup logging to see everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def check_array_health(arr: np.ndarray, name: str) -> Dict[str, Any]:
    """
    Comprehensive health check for numpy arrays
    """
    health = {
        'name': name,
        'shape': arr.shape,
        'dtype': arr.dtype,
        'min': np.min(arr),
        'max': np.max(arr),
        'mean': np.mean(arr),
        'std': np.std(arr),
        'has_nan': np.any(np.isnan(arr)),
        'has_inf': np.any(np.isinf(arr)),
        'has_negative': np.any(arr < 0),
        'has_zero': np.any(arr == 0),
        'num_nan': np.sum(np.isnan(arr)),
        'num_inf': np.sum(np.isinf(arr)),
        'num_negative': np.sum(arr < 0),
        'num_zero': np.sum(arr == 0),
        'memory_mb': arr.nbytes / 1e6
    }
    
    # Additional checks for problematic values
    if health['has_nan'] or health['has_inf']:
        health['problematic'] = True
        # Find locations of problematic values
        nan_locs = np.where(np.isnan(arr))
        inf_locs = np.where(np.isinf(arr))
        health['nan_locations'] = [tuple(loc) for loc in zip(*nan_locs)][:5]  # First 5
        health['inf_locations'] = [tuple(loc) for loc in zip(*inf_locs)][:5]  # First 5
    else:
        health['problematic'] = False
    
    return health

def print_health_report(health: Dict[str, Any]):
    """Print a readable health report"""
    print(f"\n{'='*60}")
    print(f"ARRAY HEALTH REPORT: {health['name']}")
    print(f"{'='*60}")
    print(f"Shape: {health['shape']}")
    print(f"Dtype: {health['dtype']}")
    print(f"Memory: {health['memory_mb']:.2f} MB")
    print(f"Range: [{health['min']:.6e}, {health['max']:.6e}]")
    print(f"Mean: {health['mean']:.6e}, Std: {health['std']:.6e}")
    
    print(f"\nPROBLEMATIC VALUES:")
    print(f"  NaN values: {health['num_nan']} ({'YES' if health['has_nan'] else 'NO'})")
    print(f"  Inf values: {health['num_inf']} ({'YES' if health['has_inf'] else 'NO'})")
    print(f"  Negative values: {health['num_negative']}")
    print(f"  Zero values: {health['num_zero']}")
    
    if health.get('problematic', False):
        print(f"\n⚠️  ALERT: PROBLEMATIC VALUES DETECTED!")
        if health['nan_locations']:
            print(f"  First NaN locations: {health['nan_locations']}")
        if health['inf_locations']:
            print(f"  First Inf locations: {health['inf_locations']}")
    else:
        print(f"\n✅ Array appears healthy")

def test_normalize_log_function():
    """Test the normalize_log function with various edge cases"""
    print(f"\n{'='*80}")
    print("TESTING NORMALIZE_LOG FUNCTION")
    print(f"{'='*80}")
    
    test_cases = [
        ("Normal positive data", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        ("Data with zeros", np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 5.0]])),
        ("Data with negative values", np.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])),
        ("Very small values", np.array([[1e-20, 1e-15, 1e-10], [1e-8, 1e-5, 1.0]])),
        ("Very large values", np.array([[1e10, 1e15, 1e20], [1.0, 2.0, 3.0]])),
        ("Mixed problematic", np.array([[0.0, -1e-15, np.inf], [1e20, -np.inf, np.nan]])),
        ("All zeros", np.zeros((2, 3))),
        ("All same value", np.full((2, 3), 42.0))
    ]
    
    for test_name, test_data in test_cases:
        print(f"\n--- Testing: {test_name} ---")
        print(f"Input: {test_data}")
        
        try:
            result = normalize_log(test_data)
            health = check_array_health(result, f"normalize_log({test_name})")
            print(f"Output: {result}")
            print(f"Health: NaN={health['has_nan']}, Inf={health['has_inf']}, Range=[{health['min']:.6e}, {health['max']:.6e}]")
            
            if health['problematic']:
                print(f"❌ FAILED: Produced problematic values!")
            else:
                print(f"✅ PASSED")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            traceback.print_exc()

def test_data_loading_pipeline():
    """Test the data loading and preprocessing pipeline step by step"""
    print(f"\n{'='*80}")
    print("TESTING DATA LOADING PIPELINE")
    print(f"{'='*80}")
    
    try:
        # Load config
        config = Config()
        print("✅ Config loaded successfully")
        
        # Create preprocessor
        preprocessor = DataPreprocessor(config)
        print("✅ DataPreprocessor created successfully")
        
        # Try to load a small sample of background data
        print("\n--- Loading background data sample ---")
        
        # Get first training file
        first_file = config.data.training_files[0]
        filepath = config.get_training_file_path(first_file)
        print(f"Loading from: {filepath}")
        
        # Load just first few cadences
        raw_data = np.load(filepath, mmap_mode='r')
        print(f"Raw data shape: {raw_data.shape}")
        
        # Take just first 2 cadences for testing
        sample_data = raw_data[:2].copy()  # Shape: (2, 6, 16, freq)
        health = check_array_health(sample_data, "Raw loaded data")
        print_health_report(health)
        
        if health['problematic']:
            print("❌ PROBLEM DETECTED IN RAW DATA!")
            return False
        
        # Process each cadence through the preprocessing pipeline
        all_backgrounds = []
        for cadence_idx in range(sample_data.shape[0]):
            cadence = sample_data[cadence_idx]  # Shape: (6, 16, freq)
            
            print(f"\n--- Processing cadence {cadence_idx} ---")
            
            # Check cadence health before processing
            health = check_array_health(cadence, f"Cadence {cadence_idx} input")
            if health['problematic']:
                print(f"❌ Cadence {cadence_idx} input is problematic!")
                print_health_report(health)
            
            # Reshape for preprocessing (add polarization if needed)
            observations = []
            for obs_idx in range(6):
                obs = cadence[obs_idx]  # (16, freq)
                # Add polarization dimension: (16, 2, freq)
                obs_with_pol = np.zeros((16, 2, obs.shape[1]))
                obs_with_pol[:, 0, :] = obs
                obs_with_pol[:, 1, :] = obs
                observations.append(obs_with_pol)
                
                # Check observation health
                health = check_array_health(obs_with_pol, f"Cadence {cadence_idx}, Obs {obs_idx}")
                if health['problematic']:
                    print(f"❌ Observation {obs_idx} is problematic after reshaping!")
                    print_health_report(health)
            
            # Process through preprocessor
            try:
                processed_cadence = preprocessor.preprocess_cadence(observations, use_overlap=False)
                health = check_array_health(processed_cadence, f"Processed cadence {cadence_idx}")
                print(f"Processed shape: {processed_cadence.shape}")
                
                if health['problematic']:
                    print(f"❌ PROBLEM DETECTED AFTER PREPROCESSING!")
                    print_health_report(health)
                    return False
                else:
                    print(f"✅ Cadence {cadence_idx} processed successfully")
                
                # Add to backgrounds
                for snippet_idx in range(processed_cadence.shape[0]):
                    all_backgrounds.append(processed_cadence[snippet_idx])
                    
            except Exception as e:
                print(f"❌ Error processing cadence {cadence_idx}: {e}")
                traceback.print_exc()
                return False
        
        # Create final background array
        background_array = np.array(all_backgrounds, dtype=np.float32)
        health = check_array_health(background_array, "Final background array")
        print_health_report(health)
        
        if health['problematic']:
            print("❌ FINAL BACKGROUND ARRAY IS PROBLEMATIC!")
            return False
        else:
            print("✅ Background data loading pipeline completed successfully")
            return background_array
            
    except Exception as e:
        print(f"❌ Error in data loading pipeline: {e}")
        traceback.print_exc()
        return False

def test_data_generation_pipeline(background_data):
    """Test the data generation pipeline that creates training batches"""
    print(f"\n{'='*80}")
    print("TESTING DATA GENERATION PIPELINE")
    print(f"{'='*80}")
    
    try:
        config = Config()
        
        # Create data generator
        data_generator = DataGenerator(config, background_data)
        print("✅ DataGenerator created successfully")
        
        # Generate small batches of each type
        batch_size = 4  # Small for testing
        
        for data_type in ["true", "false", "mixed", "none"]:
            print(f"\n--- Testing {data_type} data generation ---")
            
            try:
                batch_data = data_generator.generate_batch(batch_size, data_type)
                health = check_array_health(batch_data, f"Generated {data_type} batch")
                print(f"Generated shape: {batch_data.shape}")
                
                if health['problematic']:
                    print(f"❌ PROBLEM IN {data_type.upper()} BATCH GENERATION!")
                    print_health_report(health)
                    return False
                else:
                    print(f"✅ {data_type} batch generated successfully")
                    
            except Exception as e:
                print(f"❌ Error generating {data_type} batch: {e}")
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data generation pipeline: {e}")
        traceback.print_exc()
        return False

def test_model_inference_pipeline(background_data):
    """Test the VAE model with sample data to see where NaN appears"""
    print(f"\n{'='*80}")
    print("TESTING MODEL INFERENCE PIPELINE")
    print(f"{'='*80}")
    
    try:
        config = Config()
        preprocessor = DataPreprocessor(config)
        data_generator = DataGenerator(config, background_data)
        
        # Create VAE model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            strategy = tf.distribute.get_strategy()
            with strategy.scope():
                vae = create_vae_model(config)
                print("✅ VAE model created successfully")
        
        # Generate a small test batch
        print("\n--- Generating test batch ---")
        test_batch = data_generator.generate_batch(2, "true")  # Very small batch
        health = check_array_health(test_batch, "Test batch for model")
        
        if health['problematic']:
            print("❌ Test batch is problematic before model input!")
            print_health_report(health)
            return False
        
        # Prepare for model input
        print("--- Preparing batch for model ---")
        prepared_batch = preprocessor.prepare_batch(test_batch)
        health = check_array_health(prepared_batch, "Prepared batch for model")
        
        if health['problematic']:
            print("❌ Prepared batch is problematic!")
            print_health_report(health)
            return False
        
        print(f"Model input shape: {prepared_batch.shape}")
        
        # Test encoder
        print("--- Testing VAE encoder ---")
        try:
            z_mean, z_log_var, z = vae.encoder.predict(prepared_batch[:1], batch_size=1, verbose=0)  # Just 1 sample
            
            # Check each encoder output
            for name, output in [("z_mean", z_mean), ("z_log_var", z_log_var), ("z", z)]:
                health = check_array_health(output, f"Encoder {name}")
                if health['problematic']:
                    print(f"❌ PROBLEM IN ENCODER {name.upper()}!")
                    print_health_report(health)
                    return False
                else:
                    print(f"✅ Encoder {name} output healthy")
            
        except Exception as e:
            print(f"❌ Error in encoder: {e}")
            traceback.print_exc()
            return False
        
        # Test decoder
        print("--- Testing VAE decoder ---")
        try:
            decoded = vae.decoder.predict(z, batch_size=1, verbose=0)
            health = check_array_health(decoded, "Decoder output")
            
            if health['problematic']:
                print("❌ PROBLEM IN DECODER OUTPUT!")
                print_health_report(health)
                return False
            else:
                print("✅ Decoder output healthy")
            
        except Exception as e:
            print(f"❌ Error in decoder: {e}")
            traceback.print_exc()
            return False
        
        # Test full VAE
        print("--- Testing full VAE forward pass ---")
        try:
            reconstructed = vae.predict(prepared_batch[:1], batch_size=1, verbose=0)
            health = check_array_health(reconstructed, "Full VAE reconstruction")
            
            if health['problematic']:
                print("❌ PROBLEM IN FULL VAE OUTPUT!")
                print_health_report(health)
                return False
            else:
                print("✅ Full VAE forward pass healthy")
            
        except Exception as e:
            print(f"❌ Error in full VAE: {e}")
            traceback.print_exc()
            return False
        
        # Test loss computation
        print("--- Testing loss computation ---")
        try:
            # Manually compute losses like the model does
            reconstruction_loss = tf.keras.losses.mse(prepared_batch[:1], reconstructed)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            
            print(f"Reconstruction loss: {reconstruction_loss.numpy()}")
            print(f"KL loss: {kl_loss.numpy()}")
            
            if np.isnan(reconstruction_loss.numpy()) or np.isnan(kl_loss.numpy()):
                print("❌ NaN DETECTED IN LOSS COMPUTATION!")
                return False
            else:
                print("✅ Loss computation healthy")
            
        except Exception as e:
            print(f"❌ Error in loss computation: {e}")
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model inference pipeline: {e}")
        traceback.print_exc()
        return False

def main():
    """Main diagnostic routine"""
    print(f"{'='*80}")
    print("SETI ML PIPELINE NaN DIAGNOSTIC SCRIPT")
    print(f"{'='*80}")
    
    # Test 1: normalize_log function
    test_normalize_log_function()
    
    # Test 2: Data loading pipeline
    background_data = test_data_loading_pipeline()
    if background_data is False:
        print("\n❌ STOPPING: Data loading pipeline failed")
        return
    
    # Test 3: Data generation pipeline
    if not test_data_generation_pipeline(background_data):
        print("\n❌ STOPPING: Data generation pipeline failed")
        return
    
    # Test 4: Model inference pipeline
    if not test_model_inference_pipeline(background_data):
        print("\n❌ STOPPING: Model inference pipeline failed")
        return
    
    print(f"\n{'='*80}")
    print("🎉 ALL TESTS PASSED! No NaN source detected in isolated tests.")
    print("The issue may be in the training loop or distributed training setup.")
    print("Consider running with smaller batch sizes or checking optimizer settings.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
