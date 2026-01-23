"""
Debug script to see how our drawings are being preprocessed
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_preprocessing(image_path=None):
    """Visualize the preprocessing steps"""
    
    # Create a test digit (simulating our drawing)
    # White background with red digit
    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    # Draw a red '5' (simulating what user draws)
    # Using red color (0, 0, 255) in BGR
    cv2.putText(canvas, '5', (200, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 20)
    
    print("Step 1: Original canvas (BGR)")
    print(f"  Shape: {canvas.shape}")
    print(f"  Pixel at center: {canvas[250, 250]}")
    print(f"  Red channel (digit area): {canvas[250, 250, 2]}")
    print(f"  Background (white): {canvas[0, 0]}")
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    print("\nStep 2: After grayscale conversion")
    print(f"  Shape: {gray.shape}")
    print(f"  Digit area value (red->gray): {gray[250, 250]}")
    print(f"  Background value: {gray[0, 0]}")
    
    # Step 3: Invert (what we're doing)
    inverted = 255 - gray
    print("\nStep 3: After inversion (255 - gray)")
    print(f"  Digit area value: {inverted[250, 250]}")
    print(f"  Background value: {inverted[0, 0]}")
    
    # Step 4: What MNIST expects
    print("\nStep 4: MNIST original format")
    print("  MNIST has: Black background (0), White digits (255)")
    print(f"  Our result: Background = {inverted[0, 0]}, Digit = {inverted[250, 250]}")
    
    # Step 5: Alternative - don't invert, just threshold
    # Red in grayscale is ~29, we want it to be ~255 (white)
    scaled = (gray / 29.0 * 255).astype(np.uint8)  # Scale red (29) to white (255)
    scaled = np.clip(scaled, 0, 255)
    print("\nStep 5: After scaling red to white")
    print(f"  Digit area value: {scaled[250, 250]}")
    print(f"  Background value: {scaled[0, 0]}")
    
    # Step 6: Try thresholding
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    print("\nStep 6: After thresholding (THRESH_BINARY_INV, thresh=200)")
    print(f"  Digit area value: {binary[250, 250]}")
    print(f"  Background value: {binary[0, 0]}")
    
    # Visualize all steps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    titles = ['Original (BGR)', 'Grayscale', 'Inverted',
              'Scaled', 'Thresholded', 'MNIST Sample']
    
    images = [canvas, gray, inverted, scaled, binary]
    
    for i, (ax, title) in enumerate(zip(axes.flat[:-1], titles[:-1])):
        if i == 0:
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(images[i], cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    # Load actual MNIST sample for comparison
    from tensorflow import keras
    mnist = keras.datasets.mnist
    (X_train, _), _ = mnist.load_data()
    axes.flat[-1].imshow(X_train[0], cmap='gray')
    axes.flat[-1].set_title('MNIST Sample')
    axes.flat[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_debug.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print("Our drawing (red on white) -> grayscale -> inverted")
    print(f"  Gives us: Background={inverted[0, 0]}, Digit={inverted[250, 250]}")
    print("\nMNIST expects:")
    print(f"  Background=0 (black), Digit=255 (white)")
    print("\nPROBLEM: Our digits are TOO BRIGHT (226) and background is black (0)")
    print("         But MNIST digits should be mostly dark (0-50) with white (255) strokes")
    print("\nSOLUTION: We need to THRESHOLD or adjust brightness/contrast")
    print("="*60)

if __name__ == "__main__":
    visualize_preprocessing()
