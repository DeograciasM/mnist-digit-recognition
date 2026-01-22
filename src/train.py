"""
MNIST Digit Recognition - Main Training Script
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

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
    
    # One-hot encode labels for categorical crossentropy
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train_cat.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test_cat.shape}")
    
    return X_train, y_train, y_train_cat, X_test, y_test, y_test_cat

def explore_data(X_train, y_train):
    """Display basic data exploration"""
    print("\n=== Data Exploration ===")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Unique labels: {np.unique(y_train)}")
    
    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(10):
        axes[i].imshow(X_train[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis('off')
    
    plt.suptitle("Sample MNIST Images", fontsize=16)
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Sample images saved as 'sample_images.png'")
    plt.show()

def build_model():
    """Build a simple neural network model"""
    print("\n=== Building Neural Network Model ===")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(28, 28, 1)),
        
        # Flatten 28x28 images to 1D vector
        layers.Flatten(),
        
        # Hidden layers
        layers.Dense(128, activation='relu', name='hidden_layer_1'),
        layers.Dropout(0.2),  # Dropout for regularization
        
        layers.Dense(64, activation='relu', name='hidden_layer_2'),
        layers.Dropout(0.2),
        
        # Output layer (10 classes for digits 0-9)
        layers.Dense(10, activation='softmax', name='output_layer')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Display model summary
    model.summary()
    
    return model

def train_model(model, X_train, y_train_cat, X_test, y_test_cat, epochs=5):
    """Train the neural network model"""
    print("\n=== Training Model ===")
    print(f"Training for {epochs} epochs...")
    
    # Callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]
    
    # Train the model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train_cat,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\n=== Evaluating Model ===")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test, y_test_cat, verbose=0
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    return history, test_accuracy

def plot_training_history(history):
    """Plot training history (loss and accuracy curves)"""
    print("\n=== Plotting Training History ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as 'training_history.png'")
    plt.show()

def save_model(model, accuracy):
    """Save the trained model"""
    print("\n=== Saving Model ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/mnist_model_{timestamp}_acc{accuracy:.4f}.h5"
    
    model.save(model_filename)
    print(f"Model saved as: {model_filename}")
    
    # Save a lightweight version for production
    model.save('models/mnist_model_latest.h5')
    print("Latest model saved as: models/mnist_model_latest.h5")

def main():
    print("=" * 50)
    print("MNIST Digit Recognition Project")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Available: {len(gpus) > 0}")
    if gpus:
        print(f"Using GPU: {gpus[0]}")
    
    # Load data
    X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = load_data()
    
    # Explore data (optional - comment out if you don't want to see images)
    # explore_data(X_train, y_train)
    
    # Build model
    model = build_model()
    
    # Train model
    history, test_accuracy = train_model(
        model, X_train, y_train_cat, X_test, y_test_cat, epochs=5
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    save_model(model, test_accuracy)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
