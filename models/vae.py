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
    Includes robust tensor handling for distributed training
    """
    
    def __init__(self, encoder, decoder, alpha=10, beta=1.5, gamma=0, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # Hyperparameters
        self.alpha = alpha  
        self.beta = beta    
        self.gamma = gamma  
        
        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.true_loss_tracker = keras.metrics.Mean(name="true_loss")
        self.false_loss_tracker = keras.metrics.Mean(name="false_loss")

        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
        self.val_true_loss_tracker = keras.metrics.Mean(name="val_true_loss")
        self.val_false_loss_tracker = keras.metrics.Mean(name="val_false_loss")
    
    def call(self, inputs, training=None):
        """
        Forward pass through the VAE 
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
    def loss_same(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Distance between ON-ON or OFF-OFF (to be minimized)
        """
        return tf.reduce_mean(tf.reduce_sum(tf.square(a - b), axis=1))

    @tf.function
    def loss_diff(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Distance between ON-OFF (to be maximized)
        """
        return tf.reduce_mean(1.0 / (tf.reduce_sum(tf.square(a - b), axis=1) + 1e-8))

    @tf.function
    def compute_clustering_loss_true(self, true_data: tf.Tensor) -> tf.Tensor:
        """
        Clustering loss for true signals
        """
        # Add channel dimension if missing
        if len(true_data.shape) == 4:
            true_data = tf.expand_dims(true_data, -1)  # (batch, 6, 16, 512, 1)
        
        # Process each observation separately
        a1 = self.encoder(true_data[:,0,:,:,:], training=True)[2]  # ON
        b = self.encoder(true_data[:,1,:,:,:], training=True)[2]   # OFF
        a2 = self.encoder(true_data[:,2,:,:,:], training=True)[2]  # ON  
        c = self.encoder(true_data[:,3,:,:,:], training=True)[2]   # OFF
        a3 = self.encoder(true_data[:,4,:,:,:], training=True)[2]  # ON
        d = self.encoder(true_data[:,5,:,:,:], training=True)[2]   # OFF

        difference = 0.0
        difference += self.loss_diff(a1, b)
        difference += self.loss_diff(a1, c)
        difference += self.loss_diff(a1, d)
        difference += self.loss_diff(a2, b)
        difference += self.loss_diff(a2, c)
        difference += self.loss_diff(a2, d)
        difference += self.loss_diff(a3, b)
        difference += self.loss_diff(a3, c)
        difference += self.loss_diff(a3, d)
        
        same = 0.0
        same += self.loss_same(a1, a2)
        same += self.loss_same(a1, a3)
        same += self.loss_same(a2, a3)
        same += self.loss_same(b, c)
        same += self.loss_same(b, d)
        same += self.loss_same(c, d)
        
        similarity = same + difference
        return similarity

    @tf.function
    def compute_clustering_loss_false(self, false_data: tf.Tensor) -> tf.Tensor:
        """
        Clustering loss for false signals
        """
        # Add channel dimension if missing
        if len(false_data.shape) == 4:
            false_data = tf.expand_dims(false_data, -1)
        
        # Process each observation separately
        a1 = self.encoder(false_data[:,0,:,:,:], training=True)[2]  # ON
        b = self.encoder(false_data[:,1,:,:,:], training=True)[2]   # OFF
        a2 = self.encoder(false_data[:,2,:,:,:], training=True)[2]  # ON
        c = self.encoder(false_data[:,3,:,:,:], training=True)[2]   # OFF
        a3 = self.encoder(false_data[:,4,:,:,:], training=True)[2]  # ON
        d = self.encoder(false_data[:,5,:,:,:], training=True)[2]   # OFF

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
        """Model training step"""
        # Unpack data
        x, y = data
        true_data = x[1]
        false_data = x[2] 
        x = x[0]
        
        with tf.GradientTape() as tape:
            # Reshape input for encoder like in call method
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
            
            # Compute reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
                )
            )
            
            # Compute KL loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Compute clustering losses (these handle their own reshaping)
            false_loss = self.compute_clustering_loss_false(false_data)
            true_loss = self.compute_clustering_loss_true(true_data)
            
            # Total loss formula
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

    def test_step(self, data):
        """Model validation step"""
        # Unpack data same as train_step
        x, y = data
        true_data = x[1]
        false_data = x[2] 
        x = x[0]
        
        # Forward pass only (no gradient tape needed for validation)
        batch_size = tf.shape(x)[0]
        
        # Reshape input for encoder: (batch*6, 16, 512, 1)
        if len(x.shape) == 4:  # (batch, 6, 16, 512)
            encoder_input = tf.reshape(x, (batch_size * 6, 16, 512, 1))
        else:  # Already has channel dim
            encoder_input = tf.reshape(x, (batch_size * 6, 16, 512, 1))
        
        # Encode and decode
        z_mean, z_log_var, z = self.encoder(encoder_input, training=False)
        reconstruction = self.decoder(z, training=False)
        
        # Reshape reconstruction back to cadence format for loss computation
        reconstruction = tf.reshape(reconstruction, tf.shape(y))
        
        # Compute all losses exactly like train_step
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
            )
        )
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        false_loss = self.compute_clustering_loss_false(false_data)
        true_loss = self.compute_clustering_loss_true(true_data)
        
        total_loss = (reconstruction_loss + 
                     self.beta * kl_loss + 
                     self.alpha * (1 * true_loss + false_loss))

        # Update metrics
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        self.val_true_loss_tracker.update_state(true_loss)
        self.val_false_loss_tracker.update_state(false_loss)
        
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
            "true_loss": self.val_true_loss_tracker.result(),
            "false_loss": self.val_false_loss_tracker.result()
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
    """Build encoder network"""
    
    encoder_inputs = keras.Input(shape=(16, 512, 1), name="encoder_input")
    
    # Convolutional layers with regularization
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
    """Build decoder network - exact mirror of encoder"""
    
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
    
    vae.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.model.learning_rate
        )
    )
    
    logger.info(f"Created VAE model: latent_dim={config.model.latent_dim}, "
               f"beta={config.model.beta}, alpha={config.model.alpha}")
    logger.info(f"{encoder.summary()}")
    logger.info(f"{decoder.summary()}")
    
    return vae
