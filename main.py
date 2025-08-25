"""
Main entry point for SETI ML Pipeline
Fixed to properly handle data preprocessing and training flow
"""

import argparse
import logging
import os
import sys
import numpy as np
from datetime import datetime
import json
import gc

from config import Config
from preprocessing import DataPreprocessor
from training import train_full_pipeline
from inference import run_inference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/output/seti/train_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_gpu_config():
    """Configure GPU memory growth to prevent OOM"""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Configured {len(gpus)} GPUs with memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPUs detected, running on CPU")

def load_background_data(config: Config) -> np.ndarray:
    """
    Load and preprocess background observation data
    Paper: Uses 14,711 background snippets from 3 cadences
    
    Args:
        config: Configuration object with file specifications
        
    Returns:
        Preprocessed background data (n_backgrounds, 6, 16, 512)
    """
    logger.info(f"Loading background data from {config.data_path}")
    
    preprocessor = DataPreprocessor(config)
    all_backgrounds = []
    
    for filename in config.data.training_files:
        filepath = config.get_training_file_path(filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            continue
            
        logger.info(f"Processing {filename}...")
        
        # Get subset parameters
        start, end = config.get_file_subset(filename)
        
        # Load data using memory mapping to avoid OOM
        try:
            # Load raw data - expected shape: (n_cadences, 6, 16, total_freq_channels)
            raw_data = np.load(filepath, mmap_mode='r')
            
            # Apply subset if specified
            if start is not None or end is not None:
                raw_data = raw_data[start:end]
            
            logger.info(f"  Raw data shape: {raw_data.shape}")
            
            # Process each cadence
            n_cadences = raw_data.shape[0]
            for cadence_idx in range(n_cadences):
                cadence = raw_data[cadence_idx]  # Shape: (6, 16, total_freq)

                # Time gap filtering check
                # TODO: Implement when metadata with observation timestamps is available
                # For now, add a warning that we're not filtering time gaps
                if cadence_idx == 0:  # Only log once per file
                    logger.warning("Time gap filtering not implemented - assuming all cadences have <2min gaps between observations")
                # Future implementation when metadata available:
                # if observation_time_gaps and max(observation_time_gaps) >= 120:
                #     logger.info(f"Skipping cadence {cadence_idx} with {max(observation_time_gaps)}s gap")
                #     continue
                
                # Check if we need to reshape (add polarization dimension if missing)
                if len(cadence.shape) == 3 and cadence.shape[0] == 6:
                    # Shape is (6, 16, freq) - need to add polarization
                    # Assume single polarization, add dummy dimension
                    observations = []
                    for obs_idx in range(6):
                        obs = cadence[obs_idx]  # (16, freq)
                        # Add polarization dimension: (16, 2, freq)
                        obs_with_pol = np.zeros((16, 2, obs.shape[1]))
                        obs_with_pol[:, 0, :] = obs
                        obs_with_pol[:, 1, :] = obs  # Duplicate for second pol
                        observations.append(obs_with_pol)
                else:
                    # Already has correct shape
                    observations = [cadence[i] for i in range(6)]
                
                # Process cadence through preprocessor
                # This will extract snippets and downsample
                try:
                    # Process cadence through preprocessor WITH overlap for training
                    processed_cadence = preprocessor.preprocess_cadence(observations, use_overlap=True)
                    # processed_cadence shape: (n_snippets, 6, 16, 512)
                    
                    # Add each snippet as a separate background
                    for snippet_idx in range(processed_cadence.shape[0]):
                        all_backgrounds.append(processed_cadence[snippet_idx])
                        
                    if (cadence_idx + 1) % 10 == 0:
                        logger.info(f"  Processed {cadence_idx + 1}/{n_cadences} cadences")
                        
                except Exception as e:
                    logger.warning(f"  Error processing cadence {cadence_idx}: {e}")
                    continue
            
            # Clear memory
            del raw_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue
    
    if len(all_backgrounds) == 0:
        raise ValueError("No background data loaded successfully")
    
    # Stack all backgrounds
    background_array = np.array(all_backgrounds, dtype=np.float32)
    
    logger.info(f"Total background snippets loaded: {background_array.shape[0]}")
    logger.info(f"Background array shape: {background_array.shape}")
    logger.info(f"Memory usage: {background_array.nbytes / 1e9:.2f} GB")
    
    # Paper mentions 14,711 backgrounds - if we have more, subsample
    if background_array.shape[0] > 14711:
        logger.info(f"Subsampling to 14,711 backgrounds as per paper")
        indices = np.random.choice(background_array.shape[0], 14711, replace=False)
        background_array = background_array[indices]
    
    return background_array

def train_command(args):
    """Execute training command"""
    logger.info("="*60)
    logger.info("Starting SETI ML Training Pipeline")
    logger.info("="*60)
    
    # Setup GPU
    setup_gpu_config()
    
    # Load configuration
    config = Config()
    
    # Override config with command line args
    if args.epochs:
        config.training.epochs_per_round = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.rounds:
        config.training.num_training_rounds = args.rounds
    
    logger.info(f"Configuration:")
    logger.info(f"  Epochs per round: {config.training.epochs_per_round}")
    logger.info(f"  Number of rounds: {config.training.num_training_rounds}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Output path: {config.output_path}")
    
    # Load and preprocess background data
    logger.info("\nLoading background data...")
    try:
        background_data = load_background_data(config)
    except Exception as e:
        logger.error(f"Failed to load background data: {e}")
        sys.exit(1)
    
    logger.info(f"Background data loaded: {background_data.shape}")
    
    # Train models
    logger.info("\nStarting training pipeline...")
    try:
        pipeline = train_full_pipeline(
            config,
            background_data,
            n_rounds=config.training.num_training_rounds
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Save configuration
    config_path = os.path.join(config.model_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    logger.info("="*60)
    logger.info("Training completed successfully!")
    logger.info("="*60)

def inference_command(args):
    """Execute inference command"""
    logger.info("Starting inference pipeline...")
    
    # Setup GPU
    setup_gpu_config()
    
    # Load configuration
    config = Config()
    
    # Load saved config if provided
    if args.config:
        with open(args.config, 'r') as f:
            saved_config = json.load(f)
            # Update config with saved values
            for section_key, section_value in saved_config.items():
                if hasattr(config, section_key) and isinstance(section_value, dict):
                    for key, value in section_value.items():
                        if hasattr(getattr(config, section_key), key):
                            setattr(getattr(config, section_key), key, value)
    
    # Prepare observation files
    observation_files = []
    
    # Check for prepared test cadences
    test_dir = os.path.join(config.data_path, 'testing', 'prepared_cadences')
    if os.path.exists(test_dir):
        # Load prepared cadences
        for cadence_idx in range(args.n_bands):
            cadence_files = []
            for obs_idx in range(6):
                obs_file = os.path.join(test_dir, f'cadence_{cadence_idx:04d}_obs_{obs_idx}.npy')
                if os.path.exists(obs_file):
                    cadence_files.append(obs_file)
            
            if len(cadence_files) == 6:
                observation_files.append(cadence_files)
    
    if not observation_files:
        logger.error("No observation files found. Please prepare test data first.")
        sys.exit(1)
    
    logger.info(f"Found {len(observation_files)} cadences for inference")
    
    # Run inference
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = args.output or f"/output/seti/detections_{timestamp}.csv"
    
    results = run_inference(
        config,
        observation_files,
        args.vae_model,
        args.rf_model,
        output_path
    )
    
    logger.info(f"Inference completed. Results saved to {output_path}")
    
    # Print summary
    if results is not None and not results.empty:
        n_total = len(results)
        n_high_conf = len(results[results['confidence'] > 0.9])
        logger.info(f"Total detections: {n_total}")
        logger.info(f"High confidence (>90%): {n_high_conf}")

def evaluate_command(args):
    """Execute evaluation command"""
    logger.info("Starting model evaluation...")
    
    # Setup GPU
    setup_gpu_config()
    
    # Load configuration
    config = Config()
    
    # Import necessary modules
    from preprocessing import DataPreprocessor
    from data_generation import DataGenerator
    import tensorflow as tf
    from models.random_forest import RandomForestModel
    
    # Load models
    logger.info(f"Loading VAE encoder from {args.vae_model}")
    vae_encoder = tf.keras.models.load_model(args.vae_model)
    
    logger.info(f"Loading Random Forest from {args.rf_model}")
    rf_model = RandomForestModel(config)
    rf_model.load(args.rf_model)
    
    # Load or generate test data
    if args.test_data:
        logger.info(f"Loading test data from {args.test_data}")
        test_data = np.load(args.test_data, allow_pickle=True).item()
    else:
        logger.info("Generating synthetic test data...")
        # Load some background for generation
        background_data = load_background_data(config)
        generator = DataGenerator(config, background_data[:100])  # Use subset
        test_data = generator.generate_test_set()
    
    # Evaluate
    preprocessor = DataPreprocessor(config)
    
    # Prepare test data
    test_true = preprocessor.prepare_batch(test_data['true'])
    test_false = preprocessor.prepare_batch(test_data['false'])
    
    # Get predictions
    _, _, true_latents = vae_encoder.predict(test_true, batch_size=64)
    _, _, false_latents = vae_encoder.predict(test_false, batch_size=64)
    
    true_preds = rf_model.predict(true_latents)
    false_preds = rf_model.predict(false_latents)
    
    # Calculate metrics
    tpr = np.mean(true_preds == 1)
    fpr = np.mean(false_preds == 1)
    accuracy = np.mean(np.concatenate([true_preds == 1, false_preds == 0]))
    
    logger.info("="*60)
    logger.info("Evaluation Results:")
    logger.info(f"  True Positive Rate: {tpr:.3f}")
    logger.info(f"  False Positive Rate: {fpr:.3f}")
    logger.info(f"  Overall Accuracy: {accuracy:.3f}")
    logger.info("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SETI ML Pipeline - Search for ETI signals using deep learning'
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--data-path', type=str, help='Path to training data')
    train_parser.add_argument('--epochs', type=int, default=100,
                            help='Epochs per training round (default: 100)')
    train_parser.add_argument('--rounds', type=int, default=20,
                            help='Number of training rounds (default: 20)')
    train_parser.add_argument('--batch-size', type=int, default=1000,
                            help='Training batch size (default: 1000)')
    
    # Inference command
    inf_parser = subparsers.add_parser('inference', help='Run inference on data')
    inf_parser.add_argument('vae_model', type=str, help='Path to VAE encoder model')
    inf_parser.add_argument('rf_model', type=str, help='Path to Random Forest model')
    inf_parser.add_argument('--config', type=str, help='Path to saved config file')
    inf_parser.add_argument('--n-bands', type=int, default=16,
                          help='Number of frequency bands to process')
    inf_parser.add_argument('--output', type=str, help='Output file path')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('vae_model', type=str, help='Path to VAE encoder model')
    eval_parser.add_argument('rf_model', type=str, help='Path to Random Forest model')
    eval_parser.add_argument('--test-data', type=str, help='Path to test data')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'inference':
        inference_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
