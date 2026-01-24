#!/usr/bin/env python3
# Advanced Finger Drawing with MediaPipe Hand Tracking
# More accurate finger tip detection using Google's MediaPipe
# Fixed version with proper imports and error handling

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FIXED: Multiple import methods for MediaPipe compatibility
try:
    # Method 1: Standard import (most common)
    import mediapipe as mp
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        print("✓ Using MediaPipe import method 1 (standard)")
    except AttributeError:
        # Method 2: Direct module access
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions import drawing_utils as mp_drawing
        from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
        print("✓ Using MediaPipe import method 2 (direct)")
except ImportError as e:
    print(f"✗ Failed to import MediaPipe: {e}")
    print("\nPlease install MediaPipe with:")
    print("  pip install mediapipe==0.10.8")
    sys.exit(1)

# Load the trained model with fallback
print("Loading model...")
try:
    # I was trying out a few models after running the notebook and train.py
    #model = tf.keras.models.load_model('models/mnist_model_latest.h5')
    model = tf.keras.models.load_model('models/final_model.h5')
    print("✓ Model loaded successfully from models directory")
except Exception as e:
    print(f"✗ Could not load model: {e}")
    print("Creating a simple fallback model...")

    # Create a simple model as fallback
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'), # using layer.Dense from keras to specify the activation function as "relu"
        tf.keras.layers.Dense(10, activation='softmax') # using layer.Dense from keras to specify the activation function as "softmax"
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    model.save('models/mnist_model_latest.h5')
    print("✓ Fallback model created and saved")

# Initialize MediaPipe Hands
try:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe Hands initialized")
except Exception as e:
    print(f"✗ Failed to initialize MediaPipe Hands: {e}")
    sys.exit(1)

# Global variables
drawing_canvas = None
last_point = None
is_drawing = False
prediction_text = "Draw a digit with your finger!"
confidence = 0
predicted_digit = "?"
BRUSH_SIZE = 15
drawing_color = (255, 255, 255)  # White drawing (better visibility)
brush_color = (0, 255, 0)  # Green brush indicator

# Define hand landmark indices for clarity
INDEX_FINGER_TIP = 8
INDEX_FINGER_MCP = 5  # Metacarpophalangeal joint (knuckle)

def setup_drawing_canvas():
    """Initialize the drawing canvas"""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Black background
    return canvas

def get_finger_tip_coordinates(hand_landmarks, frame_shape):
    """
    Extract index finger tip coordinates from hand landmarks
    Returns (x, y) coordinates in pixel space
    """
    # Get index finger tip (landmark 8)
    h, w, _ = frame_shape

    try:
        # Try getting landmark by index
        index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
    except:
        # Fallback to attribute if available
        try:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        except:
            print("Warning: Could not get index finger tip landmark")
            return None

    # Convert normalized coordinates to pixel coordinates
    x = int(index_tip.x * w)
    y = int(index_tip.y * h)

    return (x, y)

def is_finger_raised(hand_landmarks, frame_shape):
    """Check if index finger is raised for drawing"""
    try:
        h, w, _ = frame_shape

        # Get index finger tip and MCP (knuckle)
        tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
        mcp = hand_landmarks.landmark[INDEX_FINGER_MCP]

        # Finger is raised if tip is above MCP (lower y value in image coordinates)
        return tip.y < mcp.y - 0.05  # 5% threshold
    except:
        return True  # Default to drawing if can't determine

def draw_on_canvas(point):
    """Draw on the virtual canvas"""
    global drawing_canvas, last_point

    if point is None:
        return

    x, y = point

    # Draw a circle at the current point
    cv2.circle(drawing_canvas, (x, y), BRUSH_SIZE, drawing_color, -1)

    # Draw a line from the last point to current point (for smooth drawing)
    if last_point is not None:
        cv2.line(drawing_canvas, last_point, (x, y), drawing_color, BRUSH_SIZE * 2)

    last_point = (x, y)

def predict_drawn_digit():
    """Predict the digit drawn on canvas"""
    global drawing_canvas, prediction_text, confidence, predicted_digit

    if drawing_canvas is None:
        prediction_text = "No canvas to predict!"
        return

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)

        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours to get bounding box
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            prediction_text = "No drawing detected!"
            return

        # Get bounding box of all contours combined
        all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
        x, y, w, h = cv2.boundingRect(all_points)

        if w == 0 or h == 0:
            prediction_text = "Drawing too small!"
            return

        # Add padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(drawing_canvas.shape[1], x + w + padding)
        y2 = min(drawing_canvas.shape[0], y + h + padding)

        # Crop to drawing area
        roi = binary[y1:y2, x1:x2]

        # Resize to 28x28
        roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Invert (MNIST digits are white on black background)
        roi_inverted = 255 - roi_resized

        # Normalize
        roi_normalized = roi_inverted.astype('float32') / 255.0

        # Reshape for model
        input_data = roi_normalized.reshape(1, 28, 28, 1)

        # Make prediction
        predictions = model.predict(input_data, verbose=0)
        digit = np.argmax(predictions[0])
        confidence = predictions[0][digit] * 100

        predicted_digit = str(digit)
        prediction_text = f"Prediction: {digit} ({confidence:.1f}%)"

        # Show the processed digit
        display_img = cv2.resize(roi, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Processed Digit', display_img)

        # Display confidence for all digits
        print(f"\n{'='*40}")
        print("CONFIDENCE FOR ALL DIGITS:")
        for i, conf in enumerate(predictions[0]):
            indicator = " <--" if i == digit else ""
            print(f"  Digit {i}: {conf*100:5.1f}%{indicator}")
        print(f"{'='*40}")

    except Exception as e:
        print(f"Prediction error: {e}")
        prediction_text = "Error predicting digit"
        confidence = 0

def clear_canvas():
    """Clear the drawing canvas"""
    global drawing_canvas, last_point, prediction_text, confidence, predicted_digit
    drawing_canvas = setup_drawing_canvas()
    last_point = None
    prediction_text = "Draw a digit with your finger!"
    confidence = 0
    predicted_digit = "?"

def create_display_panel(webcam_frame, drawing_frame, landmarks=None):
    """Create a combined display panel"""
    # Create a copy of webcam frame for drawing
    display_frame = webcam_frame.copy()

    # Draw hand landmarks if available
    if landmarks:
        try:
            # Try using drawing_styles if available
            if 'mp_drawing_styles' in globals():
                mp_drawing.draw_landmarks(
                    display_frame, landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            else:
                # Fallback to simple drawing
                mp_drawing.draw_landmarks(
                    display_frame, landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
        except Exception as e:
            print(f"Warning: Could not draw landmarks: {e}")

    # Add drawing canvas
    drawing_display = drawing_frame.copy()

    # Add prediction text to drawing canvas
    cv2.putText(drawing_display, prediction_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if confidence > 0:
        cv2.putText(drawing_display, f"Confidence: {confidence:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add brush size indicator
    cv2.putText(drawing_display, f"Brush: {BRUSH_SIZE}px", (10, drawing_display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Resize frames to same size
    height, width = 360, 480  # Slightly smaller for better display
    webcam_resized = cv2.resize(display_frame, (width, height))
    drawing_resized = cv2.resize(drawing_display, (width, height))

    # Create info panel
    info_panel = np.ones((100, width * 2, 3), dtype=np.uint8) * 30

    # Add instructions
    instructions = [
        "INSTRUCTIONS: Draw with index finger tip (keep finger raised)",
        "CONTROLS: 'p'=Predict  'c'=Clear  'q'=Quit  '+/-'=Brush Size"
    ]

    for i, text in enumerate(instructions):
        cv2.putText(info_panel, text, (10, 25 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

    # Add status info
    cv2.putText(info_panel, f"MediaPipe v{mp.__version__} | TensorFlow v{tf.__version__}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)

    # Combine everything
    top_row = np.hstack([webcam_resized, drawing_resized])
    combined = np.vstack([top_row, info_panel])

    return combined

def main():
    global drawing_canvas, is_drawing, BRUSH_SIZE

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize drawing canvas
    drawing_canvas = setup_drawing_canvas()

    print("\n" + "="*60)
    print("MEDIAPIPE FINGER DRAWING MNIST RECOGNITION")
    print("="*60)
    print("INSTRUCTIONS:")
    print("1. Show your hand to the webcam")
    print("2. Use your INDEX FINGER TIP to draw in the air")
    print("3. Keep finger RAISED to draw, lower to stop")
    print("4. MediaPipe will track your hand (green landmarks)")
    print("5. Press 'p' to make a prediction")
    print("6. Press 'c' to clear the drawing")
    print("7. Press 'q' to quit")
    print("8. Use '+' and '-' to adjust brush size")
    print("="*60)
    print("Note: MediaPipe provides AI-powered hand tracking!")
    print("="*60)

    # Variables for drawing control
    drawing_enabled = True
    smoothing_buffer = []
    buffer_size = 5

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip frame horizontally (mirror view)
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        finger_tip = None
        hand_landmarks = None

        if results.multi_hand_landmarks:
            # Get the first (and only) hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get index finger tip coordinates
            finger_tip = get_finger_tip_coordinates(hand_landmarks, frame.shape)

            if finger_tip:
                # Check if finger is raised for drawing
                should_draw = is_finger_raised(hand_landmarks, frame.shape)

                # Draw a circle at the finger tip with color indicating drawing state
                tip_color = (0, 255, 0) if should_draw else (0, 0, 255)  # Green for draw, red for not
                cv2.circle(frame, finger_tip, 10, tip_color, 2)
                cv2.circle(frame, finger_tip, 3, tip_color, -1)

                # Label
                label = "DRAW" if should_draw else "IDLE"
                cv2.putText(frame, label, (finger_tip[0]+15, finger_tip[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, tip_color, 2)

                # Smooth the finger tip position
                if should_draw and drawing_enabled:
                    smoothing_buffer.append(finger_tip)
                    if len(smoothing_buffer) > buffer_size:
                        smoothing_buffer.pop(0)

                    # Use weighted average for smoother drawing
                    if len(smoothing_buffer) > 0:
                        # Weight recent positions more
                        weights = np.linspace(1.0, 0.5, len(smoothing_buffer))
                        weights = weights / weights.sum()

                        avg_x = int(np.sum([p[0] * w for p, w in zip(smoothing_buffer, weights)]))
                        avg_y = int(np.sum([p[1] * w for p, w in zip(smoothing_buffer, weights)]))
                        smoothed_tip = (avg_x, avg_y)

                        # Draw on canvas
                        draw_on_canvas(smoothed_tip)
                else:
                    # Reset last point when not drawing
                    global last_point
                    last_point = None
                    smoothing_buffer.clear()

        # Create display
        display = create_display_panel(frame, drawing_canvas, hand_landmarks)

        # Calculate FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()

        # Add FPS to display
        cv2.putText(display, f"FPS: {fps}", (display.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show the display
        cv2.imshow('MediaPipe Finger Drawing', display)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            clear_canvas()
            smoothing_buffer.clear()
            print("Canvas cleared!")
        elif key == ord('p'):
            print("\nMaking prediction...")
            predict_drawn_digit()
        elif key == ord('+') or key == ord('='):
            BRUSH_SIZE = min(BRUSH_SIZE + 3, 40)
            print(f"Brush size: {BRUSH_SIZE}px")
        elif key == ord('-') or key == ord('_'):
            BRUSH_SIZE = max(BRUSH_SIZE - 3, 5)
            print(f"Brush size: {BRUSH_SIZE}px")
        elif key == ord('d'):
            # Toggle drawing mode
            drawing_enabled = not drawing_enabled
            status = "ENABLED" if drawing_enabled else "DISABLED"
            print(f"Drawing {status}")
        elif key == ord('s'):
            # Save drawing
            if drawing_canvas is not None and np.any(drawing_canvas):
                timestamp = int(time.time())
                filename = f"drawing_mediapipe_{timestamp}.png"
                cv2.imwrite(filename, drawing_canvas)
                print(f"Drawing saved as {filename}")
            else:
                print("No drawing to save")

    # Cleanup
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("THANK YOU FOR USING MEDIAPIPE FINGER DRAWING!")
    print("="*60)

if __name__ == "__main__":
    main()