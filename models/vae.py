"""
β-VAE model implementation for SETI signal detection
FIXED: Proper dimensions matching Ma et al. paper specification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1, l2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class BetaVAE(keras.Model):
    """Beta-VAE with clustering loss for SETI signal detection"""
    
    def __init__(self, encoder, decoder, alpha=10.0, beta=1.5, gamma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha  # Clustering loss weight
        self.beta = beta    # KL divergence weight  
        self.gamma = gamma  # Additional loss weight
        
        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.clustering_loss_tracker = keras.metrics.Mean(name="clustering_loss")

    def compute_clustering_loss(self, latent_vectors_list, signal_type):
        """
        Compute clustering loss for cadence observations
        
        Args:
            latent_vectors_list: List of 6 latent vectors for one cadence
            signal_type: "true" or "false"
        """
        if len(latent_vectors_list) != 6:
            return 0.0
            
        # For TRUE signals: ON observations should be similar, different from OFF
        # For FALSE signals: All observations should be similar
        
        if signal_type == "true":
            # ON observations (indices 0, 2, 4) should cluster together
            on_vectors = [latent_vectors_list[i] for i in [0, 2, 4]]
            off_vectors = [latent_vectors_list[i] for i in [1, 3, 5]]
            
            # Minimize distance within ON group
            on_loss = 0.0
            for i in range(len(on_vectors)):
                for j in range(i+1, len(on_vectors)):
                    on_loss += tf.reduce_mean(tf.square(on_vectors[i] - on_vectors[j]))
            
            # Minimize distance within OFF group  
            off_loss = 0.0
            for i in range(len(off_vectors)):
                for j in range(i+1, len(off_vectors)):
                    off_loss += tf.reduce_mean(tf.square(off_vectors[i] - off_vectors[j]))
            
            # Maximize distance between ON and OFF groups
            on_off_loss = 0.0
            for on_vec in on_vectors:
                for off_vec in off_vectors:
                    on_off_loss -= tf.reduce_mean(tf.square(on_vec - off_vec))
            
            return (on_loss + off_loss + on_off_loss) / 15.0  # Normalize by number of pairs
        
        else:  # FALSE signals - all should be similar
            total_loss = 0.0
            count = 0
            for i in range(6):
                for j in range(i+1, 6):
                    total_loss += tf.reduce_mean(tf.square(latent_vectors_list[i] - latent_vectors_list[j]))
                    count += 1
            return total_loss / count

    def train_step(self, data):
        """Custom training step with clustering loss"""
        concatenated_data, true_data, false_data = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(concatenated_data, training=True)
            reconstruction = self.decoder(z, training=True)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(concatenated_data, reconstruction),
                    axis=(1, 2)
                )
            ) / (16 * 4096)  # Normalize by dimensions
            
            # KL divergence loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Clustering losses
            true_latents = [self.encoder(tf.expand_dims(true_data[:, i, :, :], -1), training=True)[2] 
                           for i in range(6)]
            false_latents = [self.encoder(tf.expand_dims(false_data[:, i, :, :], -1), training=True)[2] 
                            for i in range(6)]
            
            true_clustering_loss = self.compute_clustering_loss(true_latents, "true")
            false_clustering_loss = self.compute_clustering_loss(false_latents, "false")
            
            # Total loss
            total_loss = (reconstruction_loss + 
                         self.beta * kl_loss +
                         self.alpha * (true_clustering_loss + false_clustering_loss))
        
        # Update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.clustering_loss_tracker.update_state(true_clustering_loss + false_clustering_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "clustering_loss": self.clustering_loss_tracker.result()
        }
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.clustering_loss_tracker
        ]

def build_encoder(latent_dim: int = 8, 
                 dense_size: int = 512,
                 kernel_size: Tuple[int, int] = (3, 3)) -> keras.Model:
    """
    Build encoder network - FIXED to match paper architecture exactly
    Input: (16, 4096, 1) - 16 time bins, 4096 frequency bins, 1 channel
    """
    
    encoder_inputs = keras.Input(shape=(16, 4096, 1))
    
    # 8 Convolutional layers with filter sizes [16, 32, 64, 128] as per paper
    # Layer 1-2: 16 filters
    x = layers.Conv2D(16, kernel_size, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(16, kernel_size, activation="relu", strides=1, padding="same")(x)
    
    # Layer 3-4: 32 filters  
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=1, padding="same")(x)
    
    # Layer 5-6: 64 filters
    x = layers.Conv2D(64, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, kernel_size, activation="relu", strides=1, padding="same")(x)
    
    # Layer 7-8: 128 filters
    x = layers.Conv2D(128, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(128, kernel_size, activation="relu", strides=1, padding="same")(x)
    
    # After 4 stride=2 operations: (16,4096) -> (1,256) approximately
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(dense_size, activation="relu",
                    activity_regularizer=l1(0.001),
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01))(x)
    
    # Latent space
    z_mean = layers.Dense(latent_dim, name="z_mean",
                         activity_regularizer=l1(0.001),
                         kernel_regularizer=l2(0.01),
                         bias_regularizer=l2(0.01))(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var",
                            activity_regularizer=l1(0.001),
                            kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01))(x)
    
    # Sampling
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def build_decoder(latent_dim: int = 8,
                 dense_size: int = 512,
                 kernel_size: Tuple[int, int] = (3, 3)) -> keras.Model:
    """
    Build decoder network - FIXED to output (16, 4096, 1)
    Must be inversely symmetrical to encoder as per paper
    """
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    # Dense layers to expand from latent space
    x = layers.Dense(dense_size, activation="relu",
                    activity_regularizer=l1(0.001),
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01))(latent_inputs)
    
    # Calculate intermediate size after encoder (1 x 256 x 128 = 32768)
    x = layers.Dense(1 * 256 * 128, activation="relu",
                    activity_regularizer=l1(0.001),
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01))(x)
    
    # Reshape to feature maps
    x = layers.Reshape((1, 256, 128))(x)
    
    # Inverse of encoder - 4 upsampling operations to get back to (16, 4096)
    
    # Upsample 1: (1,256,128) -> (2,512,128)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, kernel_size, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, kernel_size, activation="relu", padding="same")(x)
    
    # Upsample 2: (2,512,128) -> (4,1024,64)  
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, kernel_size, activation="relu", padding="same")(x)
    x = layers.Conv2D(32, kernel_size, activation="relu", padding="same")(x)
    
    # Upsample 3: (4,1024,64) -> (8,2048,32)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, kernel_size, activation="relu", padding="same")(x)
    x = layers.Conv2D(16, kernel_size, activation="relu", padding="same")(x)
    
    # Upsample 4: (8,2048,32) -> (16,4096,16)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(16, kernel_size, activation="relu", padding="same")(x)
    
    # Final output layer to get single channel
    decoder_outputs = layers.Conv2D(1, kernel_size, activation="sigmoid", padding="same")(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

def create_vae_model(config):
    """Create and compile VAE model from config"""
    
    encoder = build_encoder(
        latent_dim=config.model.latent_dim,
        dense_size=config.model.dense_layer_size,
        kernel_size=config.model.kernel_size
    )
    
    decoder = build_decoder(
        latent_dim=config.model.latent_dim,
        dense_size=config.model.dense_layer_size,
        kernel_size=config.model.kernel_size
    )
    
    vae = BetaVAE(
        encoder, decoder,
        alpha=config.model.alpha,
        beta=config.model.beta,
        gamma=config.model.gamma
    )
    
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=config.model.learning_rate))
    
    logger.info(f"Created VAE model with latent_dim={config.model.latent_dim}")
    logger.info(f"Encoder input shape: (16, 4096, 1)")
    logger.info(f"Decoder output shape: (16, 4096, 1)")
    
    return vae
