"""
Beta-VAE model implementation for SETI ML Pipeline
Fixed to match paper architecture exactly
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1, l2
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class Sampling(layers.Layer):
    """Sampling layer for VAE using reparameterization trick"""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class BetaVAE(keras.Model):
    """Beta-VAE model with custom loss functions for SETI
    Paper: β=1.5, α=10 for clustering loss
    """
    
    def __init__(self, encoder, decoder, alpha=10, beta=1.5, gamma=0, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha  # Clustering loss weight
        self.beta = beta    # KL divergence weight
        self.gamma = gamma  # Not used in paper
        
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
    def compute_clustering_loss_true(self, latent_vectors: List[tf.Tensor]) -> tf.Tensor:
        """
        Compute clustering loss for true ETI pattern
        ETI: Similar in ONs (0,2,4), different from OFFs (1,3,5)
        """
        # Unpack the 6 observations
        a1, b, a2, c, a3, d = latent_vectors
        
        # ONs should be similar to each other
        same_loss = (self.compute_similarity_loss(a1, a2) + 
                    self.compute_similarity_loss(a1, a3) +
                    self.compute_similarity_loss(a2, a3))
        
        # We want ONs to be different from OFFs
        # So, we minimize 1/distance
        # This encourages larger distances between ON-OFF pairs
        diff_loss = (self.compute_dissimilarity_loss(a1, b) +
                    self.compute_dissimilarity_loss(a1, c) +
                    self.compute_dissimilarity_loss(a1, d) +
                    self.compute_dissimilarity_loss(a2, b) +
                    self.compute_dissimilarity_loss(a2, c) +
                    self.compute_dissimilarity_loss(a2, d) +
                    self.compute_dissimilarity_loss(a3, b) +
                    self.compute_dissimilarity_loss(a3, c) +
                    self.compute_dissimilarity_loss(a3, d))
        
        return same_loss + diff_loss
    
    @tf.function
    def compute_clustering_loss_false(self, latent_vectors: List[tf.Tensor]) -> tf.Tensor:
        """
        Compute clustering loss for false RFI pattern
        RFI: Similar across all 6 observations
        """
        # All observations should be similar
        total_loss = 0.0
        for i in range(len(latent_vectors)):
            for j in range(i+1, len(latent_vectors)):
                total_loss += self.compute_similarity_loss(
                    latent_vectors[i], latent_vectors[j]
                )
        return total_loss
    
    def call(self, inputs, training=None):
        """Forward pass through VAE"""
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            # Handle case where inputs is a tuple/list of tensors
            main_input = inputs[0]
        else:
            # Handle case where inputs is a single tensor
            main_input = inputs
        
        # Handle cadence batch format: (batch_size, 6, 16, 512, 1) -> (batch_size*6, 16, 512, 1)
        input_shape = tf.shape(main_input)
        if len(main_input.shape) == 5 and main_input.shape[1] == 6:
            # Reshape from (batch_size, 6, 16, 512, 1) to (batch_size*6, 16, 512, 1)
            batch_size = input_shape[0]
            main_input = tf.reshape(main_input, (batch_size * 6, 16, 512, 1))
        elif len(main_input.shape) == 4 and main_input.shape[1] == 6:
            # Handle case without channel dimension: (batch_size, 6, 16, 512) -> (batch_size*6, 16, 512, 1)
            batch_size = input_shape[0]
            main_input = tf.reshape(main_input, (batch_size * 6, 16, 512))
            main_input = tf.expand_dims(main_input, -1)
        
        z_mean, z_log_var, z = self.encoder(main_input, training=training)
        reconstruction = self.decoder(z, training=training)
        
        # Reshape reconstruction back to cadence format if needed
        if len(inputs.shape) == 5 and inputs.shape[1] == 6:
            # Reshape back to (batch_size, 6, 16, 512, 1)
            recon_batch_size = tf.shape(inputs)[0]
            reconstruction = tf.reshape(reconstruction, (recon_batch_size, 6, 16, 512, 1))
        elif len(inputs.shape) == 4 and inputs.shape[1] == 6:
            # Reshape back to (batch_size, 6, 16, 512)
            recon_batch_size = tf.shape(inputs)[0]
            reconstruction = tf.reshape(reconstruction, (recon_batch_size, 6, 16, 512, 1))
        
        return reconstruction
    
    def train_step(self, data):
        """Custom training step with all loss components"""
        # Unpack data - restore original format
        if isinstance(data, tuple) and len(data) == 2:
            inputs, target = data
            if isinstance(inputs, tuple) and len(inputs) == 3:
                # Data format: ((concatenated_input, true_cadence, false_cadence), target)
                concatenated_input, true_cadence, false_cadence = inputs
            else:
                # Simple format: (input, target) - use input for all components
                concatenated_input = inputs
                true_cadence = inputs
                false_cadence = inputs
        else:
            # Fallback - use data as both input and target
            concatenated_input = data
            target = data
            true_cadence = data
            false_cadence = data
        
        with tf.GradientTape() as tape:
            # Forward pass on main input - reshape for encoder
            # concatenated_input shape: (batch, 6, 16, 512, 1)
            input_shape = tf.shape(concatenated_input)
            batch_size = input_shape[0]
            
            # Reshape to (batch*6, 16, 512, 1) for encoder
            encoder_input = tf.reshape(concatenated_input, (batch_size * 6, 16, 512, 1))
            z_mean, z_log_var, z = self.encoder(encoder_input, training=True)
            reconstruction = self.decoder(z, training=True)
            
            # Reshape reconstruction back to (batch, 6, 16, 512, 1)
            reconstruction = tf.reshape(reconstruction, (batch_size, 6, 16, 512, 1))
            
            # Reconstruction loss (binary crossentropy as per paper)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(target, reconstruction),
                    axis=(1, 2, 3)
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Clustering loss computation
            clustering_loss = 0.0
            if true_cadence is not None and false_cadence is not None:
                # Process true and false cadences for clustering loss
                true_shape = tf.shape(true_cadence)
                false_shape = tf.shape(false_cadence)
                true_batch_size = true_shape[0]
                false_batch_size = false_shape[0]
                
                true_latents = []
                false_latents = []
                
                for obs_idx in range(6):
                    # Extract observation from cadences
                    true_obs = true_cadence[:, obs_idx, :, :, :]  # (batch, 16, 512, 1)
                    false_obs = false_cadence[:, obs_idx, :, :, :]  # (batch, 16, 512, 1)
                    
                    # Encode observations
                    _, _, true_z = self.encoder(true_obs, training=True)
                    _, _, false_z = self.encoder(false_obs, training=True)
                    
                    true_latents.append(true_z)
                    false_latents.append(false_z)
                
                # Compute clustering losses as per paper
                true_clustering_loss = self.compute_clustering_loss_true(true_latents)
                false_clustering_loss = self.compute_clustering_loss_false(false_latents)
                clustering_loss = true_clustering_loss + false_clustering_loss
            
            # Total loss (Equation 6 from paper)
            total_loss = (reconstruction_loss + 
                         self.beta * kl_loss +
                         self.alpha * clustering_loss)
        
        # Update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.clustering_loss_tracker.update_state(clustering_loss)
        
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
    Build encoder network matching paper architecture
    Paper: 8 conv layers with filters [16,16,32,32,32,64,64,128,256]
    """
    
    encoder_inputs = keras.Input(shape=(16, 512, 1), name="encoder_input")
    
    # Convolutional layers as per paper
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
    
    logger.info(f"Built encoder with output shape: {encoder.output_shape}")
    
    return encoder

def build_decoder(latent_dim: int = 8,
                 dense_size: int = 512,
                 kernel_size: Tuple[int, int] = (3, 3)) -> keras.Model:
    """
    Build decoder network (inverse of encoder)
    """
    
    latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")
    
    # Dense layers
    x = layers.Dense(dense_size, activation="relu",
                    activity_regularizer=l1(0.001),
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01))(latent_inputs)
    
    # Calculate the size needed before reshape
    # After 4 stride-2 convolutions: 16/16=1, 512/16=32
    x = layers.Dense(1 * 32 * 256, activation="relu",
                    activity_regularizer=l1(0.001),
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01))(x)
    
    # Reshape for transposed convolutions
    x = layers.Reshape((1, 32, 256))(x)
    
    # Transposed convolutional layers (reverse of encoder)
    x = layers.Conv2DTranspose(256, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(128, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(16, kernel_size, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(16, kernel_size, activation="relu", strides=2, padding="same")(x)
    
    # Output layer with sigmoid activation
    decoder_outputs = layers.Conv2DTranspose(1, kernel_size, activation="sigmoid", padding="same")(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    logger.info(f"Built decoder with output shape: {decoder.output_shape}")
    
    return decoder

def create_vae_model(config):
    """Create and compile VAE model from config"""
    
    logger.info("Creating VAE model...")
    
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
    
    # Compile with Adam optimizer
    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.model.learning_rate)
    )
    
    # Build the model by calling it with dummy data
    logger.info("Building VAE model with dummy data...")
    import tensorflow as tf
    dummy_input = tf.zeros((1, 6, 16, 512, 1))
    _ = vae(dummy_input, training=False)
    logger.info(f"VAE model built successfully. Total parameters: {vae.count_params()}")
    
    logger.info(f"Created VAE model with latent_dim={config.model.latent_dim}, "
               f"beta={config.model.beta}, alpha={config.model.alpha}")
    
    # Print model summaries for verification
    logger.info("Encoder summary:")
    encoder.summary(print_fn=logger.info)
    logger.info("Decoder summary:")
    decoder.summary(print_fn=logger.info)
    
    return vae
