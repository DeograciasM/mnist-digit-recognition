"""
MNIST Digit Recognition - Main Training Script
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def load_data():
    """Load and prepare MNIST dataset"""
    print("Loading MNIST dataset...")
    mnist = keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape data for neural network (add channel dimension)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def explore_data(X_train, y_train, X_test):
    """Display basic data exploration"""
    print("\n=== Data Exploration ===")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of test samples: {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Unique labels: {np.unique(y_train)}")
    
    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(10):
        axes[i].imshow(X_train[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis('off')
    
    plt.suptitle("Sample MNIST Images (After Normalization)", fontsize=16)
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Sample images saved as 'sample_images.png'")
    plt.show()

def main():
    print("MNIST Digit Recognition Project")
    print("TensorFlow version:", tf.__version__)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Check if GPU is available
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Explore data
    explore_data(X_train, y_train, X_test)
    
    print("\nData loaded successfully! Ready for model building.")

if __name__ == "__main__":
    main()
