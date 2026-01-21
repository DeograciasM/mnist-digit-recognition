"""
MNIST Digit Recognition - Main Training Script
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

print("MNIST Digit Recognition Project")
print("TensorFlow version:", tf.__version__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))
