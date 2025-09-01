#!/usr/bin/env python3
"""
Test script to verify distributed training compatibility after removing tf.cond
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np

def test_distributed_training_fix():
    """Test that the VAE can train with distributed strategy without tf.cond errors"""
    
    print("=" * 60)
    print("Testing Distributed Training Fix")
    print("=" * 60)
    
    try:
        # Setup distributed strategy
        strategy = tf.distribute.MirroredStrategy()
        print(f"✓ MirroredStrategy created with {strategy.num_replicas_in_sync} replicas")
        
        with strategy.scope():
            from models.vae import BetaVAE, build_encoder, build_decoder
            
            # Create VAE within distributed scope
            encoder = build_encoder(latent_dim=8)
            decoder = build_decoder(latent_dim=8) 
            vae = BetaVAE(encoder, decoder, alpha=1.0, beta=0.5)
            
            # Compile with the same settings as training pipeline
            vae.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.0001 * 0.01,  # Ultra-conservative as in training.py
                    clipnorm=0.1,
                    epsilon=1e-7,
                    beta_1=0.8,
                    beta_2=0.99,
                    amsgrad=True
                )
            )
            
            print("✓ VAE compiled within distributed strategy")
            
            # Create a small synthetic dataset that matches expected format
            def create_synthetic_batch():
                batch_size = 32  # Per replica
                
                # Create synthetic training data in the expected format
                combined_data = tf.random.normal((batch_size, 6, 16, 512)) * 0.1
                true_data = tf.random.normal((batch_size, 6, 16, 512)) * 0.1  
                false_data = tf.random.normal((batch_size, 6, 16, 512)) * 0.1
                target = combined_data  # Autoencoder target
                
                return ((combined_data, true_data, false_data), target)
            
            # Test a single training step
            print("Testing single training step...")
            
            batch_data = create_synthetic_batch()
            
            # This should not raise the tf.cond/merge_call error anymore
            result = vae.train_on_batch(batch_data[0], batch_data[1])
            
            print(f"Training step completed successfully!")
            print(f"Loss: {result['loss']:.6f}")
            print(f"Reconstruction loss: {result['reconstruction_loss']:.6f}")
            print(f"KL loss: {result['kl_loss']:.6f}")
            print(f"Clustering loss: {result['clustering_loss']:.6f}")
            
            # Verify all losses are finite
            losses_finite = all(tf.math.is_finite(result[key]).numpy() for key in result.keys())
            
            if losses_finite:
                print("✓ All losses are finite")
            else:
                print("✗ Some losses are not finite")
                return False
                
        print("\n" + "=" * 60)
        print("✓ DISTRIBUTED TRAINING FIX SUCCESSFUL!")
        print("The tf.cond/merge_call error should be resolved.")
        print("Training should now proceed without distributed training conflicts.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error during distributed training test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing distributed training fix after removing tf.cond logic")
    
    success = test_distributed_training_fix()
    
    if success:
        print("\n🎉 Distributed training fix verified!")
        print("\nThe training should now work with MirroredStrategy without the merge_call error.")
        sys.exit(0)
    else:
        print("\n❌ Distributed training test failed.")
        sys.exit(1)