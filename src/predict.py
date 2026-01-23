"""
MNIST Digit Recognition - Prediction Script
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

def load_model():
    """Load the latest trained model"""
    try:
        model = keras.models.load_model('models/advanced_model.h5')
        print("Model loaded successfully!")
        model.summary()
        return model
    except:
        print("No trained model found. Please run train.py first.")
        return None

def make_predictions(model, X_test, y_test, num_samples=10):
    """Make predictions on test samples and visualize results"""
    print("\n=== Making Predictions ===")
    
    # Randomly select samples
    indices = random.sample(range(len(X_test)), num_samples)
    samples = X_test[indices]
    true_labels = y_test[indices]
    
    # Make predictions
    predictions = model.predict(samples, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(samples[i].reshape(28, 28), cmap='gray')
        
        # Color code based on prediction correctness
        color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
        
        axes[i].set_title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}", 
                         color=color, fontsize=10)
        axes[i].axis('off')
        
        # Add confidence score
        confidence = predictions[i][predicted_labels[i]] * 100
        axes[i].text(0.5, -0.1, f"Conf: {confidence:.1f}%", 
                    transform=axes[i].transAxes, ha='center', fontsize=8)
    
    plt.suptitle("Model Predictions on Test Samples", fontsize=16)
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.show()
    
    # Calculate accuracy on these samples
    correct = sum(predicted_labels == true_labels)
    print(f"\nPrediction Accuracy on {num_samples} samples: {correct}/{num_samples} ({correct/num_samples*100:.1f}%)")
    
    # Show confidence scores for each sample
    print("\nDetailed predictions:")
    for i in range(num_samples):
        print(f"Sample {i+1}: True={true_labels[i]}, Pred={predicted_labels[i]}, "
              f"Confidence={predictions[i][predicted_labels[i]]*100:.1f}%")

def main():
    print("MNIST Digit Recognition - Prediction Demo")
    
    # Load test data
    mnist = keras.datasets.mnist
    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test.astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Load model
    model = load_model()
    
    if model:
        # Make predictions
        make_predictions(model, X_test, y_test, num_samples=10)

if __name__ == "__main__":
    main()
