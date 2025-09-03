# Test script to verify fixes
import numpy as np
import tensorflow as tf
from config import Config
from preprocessing import DataPreprocessor
from data_generation import DataGenerator
from models.vae import create_vae_model

# Create config
config = Config()

# Create dummy background data
background = np.random.randn(100, 6, 16, 512).astype(np.float32) * 0.1 + 1.0

# Test data generation
generator = DataGenerator(config, background)
batch = generator.generate_training_batch(30)

print(f"Generated batch shapes:")
print(f"  Concatenated: {batch['concatenated'].shape}")
print(f"  True: {batch['true'].shape}")
print(f"  False: {batch['false'].shape}")

# Check for NaN
print(f"\nNaN check:")
print(f"  Concatenated has NaN: {np.any(np.isnan(batch['concatenated']))}")
print(f"  True has NaN: {np.any(np.isnan(batch['true']))}")
print(f"  False has NaN: {np.any(np.isnan(batch['false']))}")

# Test VAE creation
vae = create_vae_model(config)

# Test single training step
x = (batch['concatenated'], batch['true'], batch['false'])
y = batch['concatenated']

losses = vae.train_on_batch(x, y)
print(f"\nTraining step losses: {losses}")
print(f"Losses are finite: {all(np.isfinite(l) for l in losses.values())}")
