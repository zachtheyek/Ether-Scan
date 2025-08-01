"""
Main entry point for SETI ML Pipeline
"""

import argparse
import logging
import os
import sys
import numpy as np
from datetime import datetime
import json

from config import Config
from training import train_full_pipeline
from inference import run_inference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('seti_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_gpu_config():
    """Configure GPU memory growth"""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Configured {len(gpus)} GPUs")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")

def load_background_data(data_path: str) -> np.ndarray:
    """Load background observation data"""
    logger.info(f"Loading background data from {data_path}")
    
    # This is a placeholder - in production, this would load actual observation files
    # For now, assuming preprocessed numpy arrays
    background_files = [
        'real_filtered_LARGE_HIP110750.npy',
        'real_filtered_LARGE_HIP13402.npy',
        'real_filtered_LARGE_HIP8497.npy'
    ]
    
    backgrounds = []
    for filename in background_files:
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            data = np.load(filepath)
            backgrounds.append(data)
            logger.info(f"Loaded {filename}: shape {data.shape}")
    
    if backgrounds:
        return np.vstack(backgrounds)
    else:
        # Generate dummy data for testing
        logger.warning("No background files found, generating dummy data")
        return np.random.randn(1000, 6, 16, 4096)

def train_command(args):
    """Execute training command"""
    logger.info("Starting training pipeline...")
    
    # Setup
    setup_gpu_config()
    config = Config()
    
    # Override config with command line args
    if args.epochs:
        config.training.epochs_per_round = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    # Load background data
    background_data = load_background_data(args.data_path or config.data_path)
    
    # Train
    pipeline = train_full_pipeline(
        config,
        background_data,
        n_rounds=args.rounds
    )
    
    # Save config
    config_path = os.path.join(config.model_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info("Training completed successfully")

def inference_command(args):
    """Execute inference command"""
    logger.info("Starting inference pipeline...")
    
    # Setup
    setup_gpu_config()
    config = Config()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            saved_config = json.load(f)
            # Update config with saved values
            for key, value in saved_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Prepare observation files
    # In production, this would read actual file lists
    observation_files = []
    for i in range(args.n_bands):
        band_files = [f"band_{i}_obs_{j}.npy" for j in range(6)]
        observation_files.append(band_files)
    
    # Run inference
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = args.output or f"detections_{timestamp}.csv"
    
    results = run_inference(
        config,
        observation_files,
        args.vae_model,
        args.rf_model,
        output_path
    )
    
    logger.info(f"Inference completed. Results saved to {output_path}")

def evaluate_command(args):
    """Execute evaluation command"""
    logger.info("Starting model evaluation...")
    
    # This would implement model evaluation on test data
    # Placeholder for now
    logger.info("Evaluation command not yet implemented")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SETI ML Pipeline')
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--data-path', type=str, help='Path to training data')
    train_parser.add_argument('--epochs', type=int, help='Epochs per training round')
    train_parser.add_argument('--rounds', type=int, default=20, help='Number of training rounds')
    train_parser.add_argument('--batch-size', type=int, help='Training batch size')
    
    # Inference command
    inf_parser = subparsers.add_parser('inference', help='Run inference')
    inf_parser.add_argument('vae_model', type=str, help='Path to VAE encoder model')
    inf_parser.add_argument('rf_model', type=str, help='Path to Random Forest model')
    inf_parser.add_argument('--config', type=str, help='Path to saved config file')
    inf_parser.add_argument('--n-bands', type=int, default=16, help='Number of frequency bands')
    inf_parser.add_argument('--output', type=str, help='Output file path')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
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
