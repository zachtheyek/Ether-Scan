"""
Beta-VAE model implementation for SETI ML Pipeline
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1, l2
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class Sampling(layers.Layer):
    """Sampling layer for VAE"""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class BetaVAE(keras.Model):
    """Beta-VAE model with custom loss functions for SETI"""
    
    def __init__(self, encoder, decoder, alpha=10, beta=1.5, gamma=0, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.clustering_loss_tracker = keras.metrics.Mean(name="clustering_loss")
        
    @tf.function
    def compute_similarity_loss(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Compute Euclidean distance between latent vectors"""
        return tf.reduce_mean(tf.norm(a - b, axis=1))
    
    @tf.function
    def compute_dissimilarity_loss(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Compute inverse similarity for encouraging separation"""
        similarity = self.compute_similarity_loss(a, b)
        return 1.0 / (similarity + 1e-8)
    
    @tf.function
    def compute_clustering_loss(self, latent_vectors: List[tf.Tensor], 
                              pattern: str = "true") -> tf.Tensor:
        """
        Compute clustering loss based on expected cadence patterns
        
        Args:
            latent_vectors: List of 6 latent vectors (3 ON, 3 OFF)
            pattern: "true" for ETI signal pattern, "false" for RFI pattern
        """
        a1, b, a2, c, a3, d = latent_vectors
        
        if pattern == "true":
            # ETI signals: Similar in ONs, different from OFFs
            same_loss = (self.compute_similarity_loss(a1, a2) + 
                        self.compute_similarity_loss(a1, a3) +
                        self.compute_similarity_loss(a2, a3))
            
            diff_loss = (self.compute_dissimilarity_loss(a1, b) +
                        self.compute_dissimilarity_loss(a1, c) +
                        self.compute_dissimilarity_loss(a1, d) +
                        self.compute_dissimilarity_loss(a2, b) +
                        self.compute_dissimilarity_loss(a2, c) +
                        self.compute_dissimilarity_loss(a2, d) +
                        self.compute_dissimilarity_loss(a3, b) +
                        self.compute_dissimilarity_loss(a3, c) +
                        self.compute_dissimilarity_loss(a3, d))
        else:
            # RFI: Similar across all observations
            same_loss = (self.compute_similarity_loss(a1, b) +
                        self.compute_similarity_loss(a1, c) +
                        self.compute_similarity_loss(a1, d) +
                        self.compute_similarity_loss(a2, b) +
                        self.compute_similarity_loss(a2, c) +
                        self.compute_similarity_loss(a2, d) +
                        self.compute_similarity_loss(a3, b) +
                        self.compute_similarity_loss(a3, c) +
                        self.compute_similarity_loss(a3, d))
            
            diff_loss = 0.0
        
        return same_loss + diff_loss
    
    def call(self, inputs, training=None):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction
    
    def train_step(self, data):
        x, true_data, false_data = data[0]
        y = data[1]
        
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(x, training=True)
            reconstruction = self.decoder(z, training=True)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(y, reconstruction),
                    axis=(1, 2)
                )
            ) / (16 * 512)  # Normalize by dimensions
            
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
    """Build encoder network"""
    
    encoder_inputs = keras.Input(shape=(16, 512, 1))
    
    # Convolutional layers with increasing filters
    x = layers.Conv2D(16, kernel_size, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(16, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2D(64, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2D(128, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2D(256, kernel_size, activation="relu", strides=2, padding="same")(x)
    
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
    """Build decoder network"""
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    # Dense layers
    x = layers.Dense(dense_size, activation="relu",
                    activity_regularizer=l1(0.001),
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01))(latent_inputs)
    x = layers.Dense(1 * 32 * 256, activation="relu",
                    activity_regularizer=l1(0.001),
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01))(x)
    
    # Reshape for convolutional layers
    x = layers.Reshape((1, 32, 256))(x)
    
    # Transposed convolutional layers
    x = layers.Conv2DTranspose(256, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(128, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(16, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(16, kernel_size, activation="relu", strides=2, padding="same")(x)
    
    # Output layer
    decoder_outputs = layers.Conv2DTranspose(1, kernel_size, activation="sigmoid", padding="same")(x)
    
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
    
    return vae
