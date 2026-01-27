"""
FINAL TRAINING SCRIPT - Matches finger drawing preprocessing exactly
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def preprocess_exactly_like_drawing_app(image):
    """Preprocess images exactly like finger_draw_mediapipe.py does"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply threshold (like in finger_draw_mediapipe)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours and crop (simulate drawing detection)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get bounding box
        all_points = np.vstack([c.reshape(-1, 2) for c in contours])
        x, y, w, h = cv2.boundingRect(all_points)

        # Add padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray.shape[1], x + w + padding)
        y2 = min(gray.shape[0], y + h + padding)

        # Crop
        cropped = gray[y1:y2, x1:x2]

        # Resize to 28x28 with black background
        resized = np.zeros((28, 28), dtype=np.uint8)
        h_crop, w_crop = cropped.shape

        # Calculate scaling
        scale = min(28 / w_crop, 28 / h_crop)
        new_w, new_h = int(w_crop * scale), int(h_crop * scale)

        # Resize with aspect ratio preserved
        resized_crop = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center in 28x28
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        resized[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_crop

        # Invert (MNIST has white on black, drawing has black on white)
        inverted = 255 - resized
    else:
        # If no contours (shouldn't happen with MNIST), just use original
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        inverted = 255 - resized

    # Normalize to 0-1
    normalized = inverted.astype('float32') / 255.0

    return normalized


def build_model():

    # Building a solid model for finger-drawn digits

    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Flatten(),

        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(10, activation='softmax')
    ])

    return model


def create_augmented_dataset(X_train, y_train):
    """Create augmented dataset with variations similar to finger drawings"""
    augmented_images = []
    augmented_labels = []

    for i in range(len(X_train)):
        # Original image
        img = (X_train[i] * 255).astype(np.uint8).reshape(28, 28)

        # Add to dataset
        augmented_images.append(img)
        augmented_labels.append(y_train[i])

        # Add variations
        for _ in range(2):  # Create 2 variations per image
            # Add noise
            noisy = img.copy()
            noise = np.random.randint(-30, 30, (28, 28), dtype=np.int16)
            noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            augmented_images.append(noisy)
            augmented_labels.append(y_train[i])

            # Add blur (simulate hand movement)
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            augmented_images.append(blurred)
            augmented_labels.append(y_train[i])

            # Add rotation
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (28, 28))
            augmented_images.append(rotated)
            augmented_labels.append(y_train[i])

    return np.array(augmented_images), np.array(augmented_labels)


def main():
    print("Loading MNIST dataset...")
    mnist = keras.datasets.mnist # N1 loading the dataset from keras.datasets
    (X_train, y_train), (X_test, y_test) = mnist.load_data() # N2 splitting the dataset into train & test set


    # N3 printing the size of both the test and train dataset to better understand the split
    print(f"Original dataset size: {len(X_train)} training, {len(X_test)} test")

    # N4 Preprocessing the test data by making the relevant adjustments before testing the model otherwise we will get wrong results
    # Preprocess test set exactly like drawing app
    print("Preprocessing images...")
    X_test_processed = []
    for img in X_test:
        # Scale to 0-255 and invert (MNIST is 0-255 with white on black)
        img_255 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        processed = preprocess_exactly_like_drawing_app(img_255)
        X_test_processed.append(processed)

    X_test_processed = np.array(X_test_processed).reshape(-1, 28, 28, 1)

    # Create augmented training set
    print("Creating augmented dataset...")
    X_train_aug, y_train_aug = create_augmented_dataset(X_train, y_train)

    # Preprocess augmented training set
    X_train_processed = []
    for img in X_train_aug:
        processed = preprocess_exactly_like_drawing_app(img)
        X_train_processed.append(processed)

    X_train_processed = np.array(X_train_processed).reshape(-1, 28, 28, 1)
    y_train_aug_cat = keras.utils.to_categorical(y_train_aug, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    print(f"Augmented dataset size: {len(X_train_processed)} training samples")

    # Build model
    print("Building model...")
    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train
    print("Training model...")
    history = model.fit(
        X_train_processed, y_train_aug_cat,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_processed, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/final_model.h5')
    print("Model saved as 'models/final_model.h5'")

    # Save a lighter version for production
    model.save('models/mnist_model_latest.h5')
    print("Model saved as 'models/mnist_model_latest.h5'")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_final.png')
    plt.show()


if __name__ == "__main__":
    main()
