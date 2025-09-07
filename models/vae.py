"""
Beta-VAE model implementation for SETI ML Pipeline
Fixed to handle distributed training tensor dimension issues
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
    """
    Beta-VAE model with custom loss functions for SETI
    FIXED: Robust tensor handling for distributed training
    """
    
    def __init__(self, encoder, decoder, alpha=10, beta=1.5, gamma=0, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # Author's exact hyperparameters
        self.alpha = alpha  # Clustering loss weight = 10
        self.beta = beta    # KL divergence weight = 1.5
        self.gamma = gamma  # Score loss weight = 0 (not used)
        
        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.true_loss_tracker = keras.metrics.Mean(name="true_loss")
        self.false_loss_tracker = keras.metrics.Mean(name="false_loss")
    
    def call(self, inputs, training=None):
        """
        Forward pass through the VAE with proper input handling
        """
        # Handle different input formats
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            # Training format: (concatenated, true, false)
            main_input, true_data, false_data = inputs
        else:
            # Inference format: single tensor
            main_input = inputs
        
        # Process main input through encoder-decoder
        batch_size = tf.shape(main_input)[0]
        
        # Add channel dimension if needed and reshape for encoder: (batch*6, 16, 512, 1)
        if len(main_input.shape) == 4:  # (batch, 6, 16, 512)
            encoder_input = tf.reshape(main_input, (batch_size * 6, 16, 512, 1))
        else:  # Already has channel dimension (batch, 6, 16, 512, 1)
            encoder_input = tf.reshape(main_input, (batch_size * 6, 16, 512, 1))
        
        # Encode
        z_mean, z_log_var, z = self.encoder(encoder_input, training=training)
        
        # Decode
        reconstruction = self.decoder(z, training=training)
        
        # Reshape back to cadence format: (batch, 6, 16, 512)
        reconstruction = tf.reshape(reconstruction, (batch_size, 6, 16, 512))
        
        return reconstruction
    
    @tf.function
    def safe_reduce_mean(self, tensor, axis=None):
        """Safely reduce mean with proper axis handling"""
        if axis is not None:
            # Check if the tensor has enough dimensions
            tensor_rank = len(tensor.shape)
            if isinstance(axis, int):
                if axis >= tensor_rank or axis < -tensor_rank:
                    # If axis is invalid, just return the mean of the entire tensor
                    return tf.reduce_mean(tensor)
            return tf.reduce_mean(tensor, axis=axis)
        return tf.reduce_mean(tensor)
    
    @tf.function
    def safe_reduce_sum(self, tensor, axis=None):
        """Safely reduce sum with proper axis handling"""
        if axis is not None:
            # Check if the tensor has enough dimensions
            tensor_rank = len(tensor.shape)
            if isinstance(axis, int):
                if axis >= tensor_rank or axis < -tensor_rank:
                    # If axis is invalid, just return the sum of the entire tensor
                    return tf.reduce_sum(tensor)
            return tf.reduce_sum(tensor, axis=axis)
        return tf.reduce_sum(tensor)
    
    @tf.function
    def compute_reconstruction_loss(self, target, reconstruction):
        """
        Compute reconstruction loss with robust tensor handling
        FIXED: Handles distributed training edge cases
        """
        batch_size = tf.shape(target)[0]
        
        # Flatten tensors for loss computation
        target_flat = tf.reshape(target, (batch_size, -1))
        reconstruction_flat = tf.reshape(reconstruction, (batch_size, -1))
        
        # Compute binary crossentropy
        bce = keras.losses.binary_crossentropy(target_flat, reconstruction_flat)
        
        # Handle different tensor shapes that can occur in distributed training
        if len(bce.shape) == 0:
            # Already a scalar
            return bce
        elif len(bce.shape) == 1:
            # Vector of losses per sample
            return tf.reduce_mean(bce)
        else:
            # Higher dimensional - reduce along all but first axis
            return self.safe_reduce_mean(self.safe_reduce_sum(bce, axis=1))
    
    @tf.function
    def loss_same(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Author's EXACT loss_same implementation
        """
        return tf.reduce_mean(tf.reduce_sum(tf.square(a - b), axis=1))

    @tf.function
    def compute_clustering_loss_true(self, true_data: tf.Tensor) -> tf.Tensor:
        """
        EXACT author's implementation - processes each observation separately
        Author's true_clustering function
        """
        # Add channel dimension if missing
        if len(true_data.shape) == 4:
            true_data = tf.expand_dims(true_data, -1)  # (batch, 6, 16, 512, 1)
        
        # Process each observation separately as author does
        a1 = self.encoder(true_data[:,0,:,:,:], training=True)[2]  # ON
        b = self.encoder(true_data[:,1,:,:,:], training=True)[2]   # OFF
        a2 = self.encoder(true_data[:,2,:,:,:], training=True)[2]  # ON  
        c = self.encoder(true_data[:,3,:,:,:], training=True)[2]   # OFF
        a3 = self.encoder(true_data[:,4,:,:,:], training=True)[2]  # ON
        d = self.encoder(true_data[:,5,:,:,:], training=True)[2]   # OFF

        # Author's EXACT computation (only uses loss_same, no loss_diff!)
        same = 0.0
        same += self.loss_same(a1, a2)
        same += self.loss_same(a1, a3)
        same += self.loss_same(a2, a1)
        same += self.loss_same(a2, a3)
        same += self.loss_same(a3, a2)
        same += self.loss_same(a3, a1)
        
        same += self.loss_same(b, c)
        same += self.loss_same(b, d)
        same += self.loss_same(c, b)
        same += self.loss_same(c, d)
        same += self.loss_same(d, b)
        same += self.loss_same(d, c)
        
        # Author uses only similarity distances, no explosive dissimilarity
        difference = 0.0
        difference += self.loss_same(a1, b)
        difference += self.loss_same(a1, c)
        difference += self.loss_same(a1, d)
        difference += self.loss_same(a2, b)
        difference += self.loss_same(a2, c)
        difference += self.loss_same(a2, d)
        difference += self.loss_same(a3, b)
        difference += self.loss_same(a3, c)
        difference += self.loss_same(a3, d)
        
        similarity = same + difference
        return similarity

    @tf.function
    def compute_clustering_loss_false(self, false_data: tf.Tensor) -> tf.Tensor:
        """
        EXACT author's false_clustering implementation
        """
        # Add channel dimension if missing
        if len(false_data.shape) == 4:
            false_data = tf.expand_dims(false_data, -1)
        
        # Process each observation separately
        a1 = self.encoder(false_data[:,0,:,:,:], training=True)[2]
        b = self.encoder(false_data[:,1,:,:,:], training=True)[2]
        a2 = self.encoder(false_data[:,2,:,:,:], training=True)[2]
        c = self.encoder(false_data[:,3,:,:,:], training=True)[2]
        a3 = self.encoder(false_data[:,4,:,:,:], training=True)[2]
        d = self.encoder(false_data[:,5,:,:,:], training=True)[2]

        # Author's approach - all observations should be similar for RFI
        difference = 0.0
        difference += self.loss_same(a1, b)
        difference += self.loss_same(a1, c)
        difference += self.loss_same(a1, d)
        difference += self.loss_same(a2, b)
        difference += self.loss_same(a2, c)
        difference += self.loss_same(a2, d)
        difference += self.loss_same(a3, b)
        difference += self.loss_same(a3, c)
        difference += self.loss_same(a3, d)

        same = 0.0
        same += self.loss_same(a1, a2)
        same += self.loss_same(a1, a3)
        same += self.loss_same(a2, a3)
        same += self.loss_same(b, c)
        same += self.loss_same(c, d)
        same += self.loss_same(b, d)
        
        similarity = same + difference
        return similarity
    
    def train_step(self, data):
        """Fixed train_step with proper input reshaping"""
        # Author's exact data unpacking
        x, y = data
        true_data = x[1]
        false_data = x[2] 
        x = x[0]
        
        with tf.GradientTape() as tape:
            # CRITICAL FIX: Reshape input for encoder like in call method
            batch_size = tf.shape(x)[0]
            
            # Add channel dimension and reshape for encoder: (batch*6, 16, 512, 1)
            if len(x.shape) == 4:  # (batch, 6, 16, 512)
                encoder_input = tf.reshape(x, (batch_size * 6, 16, 512, 1))
            else:  # Already has channel dim
                encoder_input = tf.reshape(x, (batch_size * 6, 16, 512, 1))
            
            # Encode
            z_mean, z_log_var, z = self.encoder(encoder_input, training=True)
            
            # Decode
            reconstruction = self.decoder(z, training=True)
            
            # Reshape reconstruction back to cadence format for loss computation
            reconstruction = tf.reshape(reconstruction, tf.shape(y))
            
            # Author's exact reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
                )
            )
            
            # Author's exact KL loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Author's clustering losses (these handle their own reshaping)
            false_loss = self.compute_clustering_loss_false(false_data)
            true_loss = self.compute_clustering_loss_true(true_data)
            
            # Author's exact total loss formula
            total_loss = (reconstruction_loss + 
                         self.beta * kl_loss + 
                         self.alpha * (1 * true_loss + false_loss))
        
        # Apply gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.true_loss_tracker.update_state(true_loss)
        self.false_loss_tracker.update_state(false_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "true_loss": self.true_loss_tracker.result(),
            "false_loss": self.false_loss_tracker.result()
        }
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.true_loss_tracker,
            self.false_loss_tracker
        ]

def build_encoder(latent_dim: int = 8, 
                 dense_size: int = 512,
                 kernel_size: Tuple[int, int] = (3, 3)) -> keras.Model:
    """
    Build encoder network matching paper architecture exactly
    CRITICAL: Includes all regularization from author's code
    """
    
    encoder_inputs = keras.Input(shape=(16, 512, 1), name="encoder_input")
    
    # Convolutional layers with author's exact regularization
    x = layers.Conv2D(16, kernel_size, activation="relu", strides=2, padding="same",
                      activity_regularizer=l1(0.001),
                      kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(encoder_inputs)
    
    x = layers.Conv2D(16, kernel_size, activation="relu", strides=1, padding="same",
                      activity_regularizer=l1(0.001),
                      kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=2, padding="same",
                      activity_regularizer=l1(0.001),
                      kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=1, padding="same",
                      activity_regularizer=l1(0.001),
                      kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=1, padding="same",
                      activity_regularizer=l1(0.001),
                      kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2D(64, kernel_size, activation="relu", strides=2, padding="same",
                      activity_regularizer=l1(0.001),
                      kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2D(64, kernel_size, activation="relu", strides=1, padding="same",
                      activity_regularizer=l1(0.001),
                      kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2D(128, kernel_size, activation="relu", strides=1, padding="same",
                      activity_regularizer=l1(0.001),
                      kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2D(256, kernel_size, activation="relu", strides=2, padding="same",
                      activity_regularizer=l1(0.001),
                      kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01))(x)
    
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
    Build decoder network - exact mirror of encoder
    """
    
    latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")
    
    # Dense layers with regularization
    x = layers.Dense(dense_size, activation="relu",
                    activity_regularizer=l1(0.001),
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01))(latent_inputs)
    
    # Reshape to start transposed convolutions
    x = layers.Dense(1 * 32 * 256, activation="relu",
                    activity_regularizer=l1(0.001),
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01))(x)
    
    x = layers.Reshape((1, 32, 256))(x)
    
    # Transposed convolutions (exact reverse of encoder)
    x = layers.Conv2DTranspose(256, kernel_size, activation="relu", strides=2, padding="same",
                              activity_regularizer=l1(0.001),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2DTranspose(128, kernel_size, activation="relu", strides=1, padding="same",
                              activity_regularizer=l1(0.001),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2DTranspose(64, kernel_size, activation="relu", strides=1, padding="same",
                              activity_regularizer=l1(0.001),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2DTranspose(64, kernel_size, activation="relu", strides=2, padding="same",
                              activity_regularizer=l1(0.001),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=1, padding="same",
                              activity_regularizer=l1(0.001),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=1, padding="same",
                              activity_regularizer=l1(0.001),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=2, padding="same",
                              activity_regularizer=l1(0.001),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2DTranspose(16, kernel_size, activation="relu", strides=1, padding="same",
                              activity_regularizer=l1(0.001),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(x)
    
    x = layers.Conv2DTranspose(16, kernel_size, activation="relu", strides=2, padding="same",
                              activity_regularizer=l1(0.001),
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01))(x)
    
    # Output layer with sigmoid activation
    decoder_outputs = layers.Conv2DTranspose(1, kernel_size, activation="sigmoid", padding="same")(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    return decoder

def create_vae_model(config):
    """Create and compile VAE model with author's exact settings"""
    
    logger.info("Creating VAE model...")
    
    encoder = build_encoder(
        latent_dim=config.model.latent_dim,  # 8
        dense_size=config.model.dense_layer_size,  # 512
        kernel_size=config.model.kernel_size  # (3, 3)
    )
    
    decoder = build_decoder(
        latent_dim=config.model.latent_dim,
        dense_size=config.model.dense_layer_size,
        kernel_size=config.model.kernel_size
    )
    
    vae = BetaVAE(
        encoder, decoder,
        alpha=config.model.alpha,  # 10
        beta=config.model.beta,    # 1.5
        gamma=config.model.gamma   # 0
    )
    
    # Compile with author's optimizer settings
    vae.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.model.learning_rate  # 0.001
        )
    )
    
    logger.info(f"Created VAE model: latent_dim={config.model.latent_dim}, "
               f"beta={config.model.beta}, alpha={config.model.alpha}")
    
    return vae
