import os
import time
import cv2
import numpy as np
import tensorflow as tf
import serial

# ==========================================
# SETTINGS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "fruit_quality_model.keras"
)

ARDUINO_PORT = "COM3"  # Change if your Arduino uses another port
BAUD_RATE = 9600

IMAGE_SIZE = (180, 180)
CONFIDENCE_THRESHOLD = 70.0

# Must match the class order used during training.
# Standard inferred Keras folder order is alphabetical.
CLASS_NAMES = ["defective", "good", "rotten"]


# ==========================================
# LOAD AI MODEL
# ==========================================

print("Loading AI model...")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("AI model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        "Model output count does not match CLASS_NAMES."
    )


# ==========================================
# CONNECT TO ARDUINO
# ==========================================

arduino = None
cap = None

try:
    print(f"Connecting to Arduino on {ARDUINO_PORT}...")

    arduino = serial.Serial(
        ARDUINO_PORT,
        BAUD_RATE,
        timeout=1
    )

    # Arduino boards may reset when serial opens.
    time.sleep(2)
    print("Arduino connected.")

    # ==========================================
    # START CAMERA
    # ==========================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Camera could not be opened. "
            "Check camera permissions or try camera index 1."
        )

    print("Camera started.")
    print("Show one fruit at a time.")
    print("Press Q to quit.")

    last_result = None

    # ==========================================
    # MAIN LOOP
    # ==========================================

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            break

        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Resize and prepare model input
        img = cv2.resize(rgb_frame, IMAGE_SIZE)
        img = np.asarray(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)

        # ==========================================
        # AI PREDICTION
        # ==========================================

        predictions = model.predict(img, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index]) * 100

        predicted_class = CLASS_NAMES[predicted_index]

        # ==========================================
        # DETERMINE RESULT
        # ==========================================

        if confidence < CONFIDENCE_THRESHOLD:
            result = "UNCERTAIN"
            display_text = f"UNCERTAIN ({confidence:.1f}%)"

        elif predicted_class == "good":
            result = "GOOD"
            display_text = f"GOOD - EXPORT ({confidence:.1f}%)"

        elif predicted_class == "defective":
            result = "DEFECTIVE"
            display_text = f"DEFECTIVE ({confidence:.1f}%)"

        else:
            result = "ROTTEN"
            display_text = f"ROTTEN - REJECT ({confidence:.1f}%)"

        # ==========================================
        # SEND RESULT TO ARDUINO
        # ==========================================

        if result != last_result:
            command = f"{result},{confidence:.1f}\n"

            try:
                arduino.write(command.encode("utf-8"))
                print("Sent to Arduino:", command.strip())
                last_result = result

            except serial.SerialException as error:
                print("Arduino communication error:", error)
                break

        # ==========================================
        # DISPLAY RESULT
        # ==========================================

        if result == "GOOD":
            color = (0, 255, 0)
        elif result == "UNCERTAIN":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.putText(
            frame,
            display_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow(
            "AI Fruit Quality Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # ==========================================
    # CLEANUP
    # ==========================================

    if cap is not None:
        cap.release()

    if arduino is not None and arduino.is_open:
        arduino.close()

    cv2.destroyAllWindows()
    print("Camera and Arduino connection closed.")