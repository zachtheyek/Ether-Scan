"""
Validate trained models are saved & loaded properly
"""

import tensorflow as tf
import joblib
import json

# Load VAE encoder
encoder = tf.keras.models.load_model('/datax/scratch/zachy/models/etherscan/vae_encoder_final_v1.keras')
print(f"Encoder loaded: {encoder.summary()}")

# Load VAE decoder
decoder = tf.keras.models.load_model('/datax/scratch/zachy/models/etherscan/vae_decoder_final_v1.keras')
print(f"Decoder loaded: {decoder.summary()}")

# Load Random Forest
rf = joblib.load('/datax/scratch/zachy/models/etherscan/random_forest_final_v1.joblib')
print(f"Random Forest loaded: {rf.n_estimators} trees")

# Load config
with open('/datax/scratch/zachy/models/etherscan/config_final_v1.json', 'r') as f:
    config = json.load(f)
print(f"Config loaded: {config.keys()}")
