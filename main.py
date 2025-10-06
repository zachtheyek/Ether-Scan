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
import gc
import time
from skimage.transform import downscale_local_mean

from config import Config
from preprocessing import DataPreprocessor, pre_proc
from data_generation import DataGenerator
from training import train_full_pipeline, get_latest_tag
from inference import run_inference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/datax/scratch/zachy/outputs/etherscan/train_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_gpu_config():
    """Configure GPU memory growth, memory limits, multi-GPU strategy with load balancing & async allocator"""
    import tensorflow as tf
    
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Prevent memory fragmentation within each GPU
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'  # Aggressive cleanup of intermediate tensors

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set equal memory limits for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14000)]  # 14GiB limit per GPU
                )

            # Set distributed strategy to prevent uneven GPU memory usage
            try:
                # Primary choice: NCCL for NVIDIA GPUs
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.NcclAllReduce()
                )
                logger.info("Using NcclAllReduce for optimal NVIDIA GPU performance")
                
            except Exception as e:
                # Fallback: HierarchicalCopyAllReduce
                logger.warning(f"NCCL failed ({e}), using HierarchicalCopyAllReduce")
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
                )
            
            logger.info(f"Distributed strategy: {strategy.num_replicas_in_sync} GPUs")
            return strategy
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
            return None
    else:
        logger.warning("No GPUs detected, running on CPU")
        return None

def load_background_data(config: Config) -> np.ndarray:
    """
    Load, downsample, and normalize background data
    """
    logger.info(f"Loading background data from {config.data_path}")
    
    # Use config values for memory management
    target_backgrounds = config.training.target_backgrounds
    chunk_size = config.data.chunk_size_loading
    max_chunks = config.data.max_chunks_per_file
    downsample_factor = config.data.downsample_factor
    final_width = config.data.width_bin // downsample_factor
    
    logger.info(f"Target backgrounds: {target_backgrounds}")
    logger.info(f"Processing chunks of: {chunk_size}")
    logger.info(f"Final resolution: {final_width}")
    
    all_backgrounds = []
    
    for filename in config.data.training_files:
        filepath = config.get_training_file_path(filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            continue
            
        logger.info(f"Processing {filename}...")
        
        # Get subset parameters from config
        start, end = config.get_file_subset(filename)
        
        try:
            # Use memory mapping to avoid loading full file
            raw_data = np.load(filepath, mmap_mode='r')
            
            # Apply subset if specified in config
            if start is not None or end is not None:
                raw_data = raw_data[start:end]
            
            logger.info(f"  Raw data shape: {raw_data.shape}")
            
            # Process in config-specified chunks
            n_chunks = min(max_chunks, (raw_data.shape[0] + chunk_size - 1) // chunk_size)
            
            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, raw_data.shape[0])
                
                # Load chunk into memory
                chunk_data = np.array(raw_data[chunk_start:chunk_end])
                
                # Process each cadence in chunk
                for cadence_idx in range(chunk_data.shape[0]):
                    if len(all_backgrounds) >= target_backgrounds:
                        break
                        
                    cadence = chunk_data[cadence_idx]  # Shape: (6, 16, 4096)
                    
                    # Skip invalid cadences
                    if np.any(np.isnan(cadence)) or np.any(np.isinf(cadence)) or np.max(cadence) <= 0:
                        continue
                    
                    # Downsample & normalize each observation separately
                    downsampled_cadence = np.zeros((6, 16, final_width), dtype=np.float32)
                    
                    for obs_idx in range(6):
                        # 1. Downsample first
                        downsampled_obs = downscale_local_mean(
                            cadence[obs_idx], (1, downsample_factor)
                        ).astype(np.float32)
                        
                        # 2. Normalize each observation using pre_proc
                        downsampled_cadence[obs_idx] = pre_proc(downsampled_obs)
                    
                    all_backgrounds.append(downsampled_cadence)
                
                # Clear chunk from memory
                del chunk_data
                gc.collect()
                
                if len(all_backgrounds) >= target_backgrounds:
                    break
            
            logger.info(f"  Processed {len(all_backgrounds)} cadences so far")
            
            # Clear raw data reference
            del raw_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue
    
    if len(all_backgrounds) == 0:
        raise ValueError("No background data loaded successfully")
    
    # Stack all backgrounds
    background_array = np.array(all_backgrounds, dtype=np.float32)
    
    # Verify that values are now in [0,1] range
    min_val = np.min(background_array)
    max_val = np.max(background_array)
    mean_val = np.mean(background_array)
    
    logger.info(f"Total background cadences loaded: {background_array.shape[0]}")
    logger.info(f"Background array shape: {background_array.shape}")
    logger.info(f"Background value range: [{min_val:.6f}, {max_val:.6f}]")
    logger.info(f"Background mean: {mean_val:.6f}")
    logger.info(f"Memory usage: {background_array.nbytes / 1e9:.2f} GB")
    
    if max_val > 2.0:
        logger.error(f"Background values still too large! Max: {max_val}")
        raise ValueError("Background normalization failed")
    else:
        logger.info(f"Background data properly normalized")
    
    logger.info(f"Background data ready at {final_width} resolution")
    
    return background_array

def train_command(args):
    """Execute training command with distributed strategy"""
    logger.info("="*60)
    logger.info("Starting SETI ML Training Pipeline")
    logger.info("="*60)
    
    # Setup GPU and get strategy
    strategy = setup_gpu_config()
    
    # Load configuration
    config = Config()
    
    # Override config with command line args
    if args.rounds:
        config.training.num_training_rounds = args.rounds
    if args.epochs:
        config.training.epochs_per_round = args.epochs
    if args.batch_size:
        config.training.train_logical_batch_size = args.batch_size
    if args.load_tag:
        tag = args.load_tag
        if tag.startswith('round_'):
            start_round = int(tag.split('_')[1]) + 1 # Start training from the round proceeding model checkpoint
        else:
            start_round = 1
    else: 
        tag = None
        start_round = 1
    if args.load_dir:
        dir = args.load_dir
    else: 
        dir = None
    
    logger.info(f"Configuration:")
    logger.info(f"  Number of rounds: {config.training.num_training_rounds}")
    logger.info(f"  Epochs per round: {config.training.epochs_per_round}")
    logger.info(f"  Batch size: {config.training.train_logical_batch_size}")
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

    # Train models with fault tolerance
    logger.info("\nStarting training pipeline...")

    max_retries = config.training.max_retries
    retry_delay = config.training.retry_delay

    for attempt in range(max_retries):
        try:
            logger.info(f"Training attempt: {attempt+1}/{max_retries}")

            if attempt > 0:
                logger.info(f"Retrying training from round {start_round}")

            # Reinitialize training pipeline on each attempt so no corrupted state is persisted
            pipeline = train_full_pipeline(
                config,
                background_data,
                strategy=strategy,
                tag=tag,
                dir=dir,
                start_round=start_round
            )

            # If we get here, training succeeded
            break

        except KeyboardInterrupt:
            # Don't retry on user interruption 
            logger.info("Training interupted by user")
            raise 

        except Exception as e:
            logger.error(f"Training attempt {attempt+1} failed with error: {e}")

            if attempt < max_retries - 1:
                # Retry taining after delay
                logger.info(f"Attempting to recover from failure: attempt {attempt+2}/{max_retries}")

                try:
                    # Find the latest checkpoint & determine where to resume from
                    dir = 'checkpoints'
                    tag = get_latest_tag(os.path.join(config.model_path, dir))
                    if tag.startswith("round_"):
                        start_round = int(tag.split("_")[1]) + 1  # Start training from the round proceeding model checkpoint
                        logger.info(f"Loaded latest checkpoint from round {start_round-1}")
                    else:
                        logger.info("No valid checkpoints loaded")
                        raise
                    
                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                
                except Exception as recovery_error:
                    # If no checkpoints loaded, restart from last valid start_round
                    logger.error(f"Recovery failed: {recovery_error}")
                    logger.info(f"Restarting training from round {start_round} in {retry_delay} seconds")
                    time.sleep(retry_delay)

            else:
                # Max retries exceeded
                logger.error(f"Training attempts exceeded maximum retries ({max_retries})")
                logger.error(f"Final error: {e}")
                raise Exception(f"Training attempts exceeded maximum retries ({max_retries}). Final error: {e}")
    
    # Save configuration
    config_path = os.path.join(config.model_path, 'config_final_v1.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    logger.info("="*60)
    logger.info("Training completed successfully!")
    logger.info("="*60)

# NOTE: come back to this later
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
    output_path = args.output or f"/outputs/seti/detections_{timestamp}.csv"
    
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

# NOTE: come back to this later
def evaluate_command(args):
    """Execute evaluation command"""
    logger.info("Starting model evaluation...")
    
    # Setup GPU
    setup_gpu_config()
    
    # Load configuration
    config = Config()
    
    # Import necessary modules
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

# TODO: add assertions to make sure no problematic values gets passed through CLI args
# NOTE: come back to this later
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SETI ML Pipeline - Search for ETI signals using deep learning'
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--rounds', type=int, default=None,
                            help='Number of training rounds (default: 20)')
    train_parser.add_argument('--epochs', type=int, default=None,
                            help='Epochs per training round (default: 100)')
    train_parser.add_argument('--batch-size', type=int, default=None,
                            help='Training batch size (default: 32)')
    train_parser.add_argument('--load-tag', type=str, default=None,
                              help='Model tag to resume training from (accepted formats: final_vX, round_XX, YYYYMMDD_HHMMSS)')
    train_parser.add_argument('--load-dir', type=str, default=None,
                            help='Directory to load model tag from (loads latest tag if none provided)')
    # train_parser.add_argument('--start-round', type=int, default=None,
                            # help='Training round to start from (default: 1, or the next round proceeding checkpoint tag if provided)')
    # train_parser.add_argument('--save-tag', type=str, default=None,
    #                         help='Model tag to save final model (recommended: final_vX)')

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
