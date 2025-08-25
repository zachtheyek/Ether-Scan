"""
Validate trained models are saved & loaded properly
"""

import tensorflow as tf
import joblib
import json

# Load VAE encoder
encoder = tf.keras.models.load_model('/models/seti/vae_encoder_final.h5')
print(f"Encoder loaded: {encoder.summary()}")

# Load Random Forest
rf = joblib.load('/models/seti/random_forest_final.joblib')
print(f"Random Forest loaded: {rf.n_estimators} trees")

# Load config
with open('/models/seti/config.json', 'r') as f:
    config = json.load(f)
print(f"Config loaded: {config.keys()}")
