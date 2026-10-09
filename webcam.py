import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\malli\OneDrive\Desktop\Fruit_Quality_AI\fruit_quality_model.keras"

# CHANGE COM PORT IF NEEDED
ARDUINO_PORT = "COM3"

BAUD_RATE = 9600

# Prediction confidence
CONFIDENCE_THRESHOLD = 0.70

# Number of consecutive similar predictions required
STABLE_FRAMES = 8

# Time before another apple can be detected
COOLDOWN_TIME = 5


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "defective",
    "good",
    "rotten"
]

print("Model loaded successfully.")


# ============================================================
# CONNECT TO ARDUINO
# ============================================================

try:

    arduino = serial.Serial(
        ARDUINO_PORT,
        BAUD_RATE,
        timeout=1
    )

    time.sleep(2)

    print("Arduino connected:", ARDUINO_PORT)

except Exception as e:

    print("ERROR: Could not connect to Arduino.")
    print(e)

    arduino = None


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")
    exit()


print("Camera started.")
print("Press Q to quit.")


# ============================================================
# VARIABLES
# ============================================================

previous_prediction = None
stable_count = 0

apple_processed = False
last_processed_time = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read camera frame.")
        break


    # --------------------------------------------------------
    # OpenCV uses BGR
    # Training images are RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Resize to model input size

    img = cv2.resize(
        rgb_frame,
        (180, 180)
    )


    # Convert to float

    img = img.astype(np.float32) / 255.0


    # Add batch dimension

    img = np.expand_dims(
        img,
        axis=0
    )


    # --------------------------------------------------------
    # AI PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        img,
        verbose=0
    )[0]


    class_index = np.argmax(prediction)

    confidence = prediction[class_index]

    label = class_names[class_index]


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    text = f"{label.upper()} - {confidence * 100:.1f}%"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "Fruit Quality AI",
        frame
    )


    # --------------------------------------------------------
    # STABLE PREDICTION CHECK
    # --------------------------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        if label == previous_prediction:

            stable_count += 1

        else:

            previous_prediction = label
            stable_count = 1


    else:

        stable_count = 0
        previous_prediction = None


    # --------------------------------------------------------
    # PROCESS ONE APPLE
    # --------------------------------------------------------

    if stable_count >= STABLE_FRAMES:

        current_time = time.time()


        # Prevent repeated detection of same apple

        if (
            not apple_processed
            and
            current_time - last_processed_time > COOLDOWN_TIME
        ):

            print()
            print("--------------------------------")
            print("APPLE DETECTED")
            print("RESULT:", label.upper())
            print(
                "CONFIDENCE:",
                f"{confidence * 100:.2f}%"
            )
            print("--------------------------------")


            # ------------------------------------------------
            # SEND RESULT TO ARDUINO
            # ------------------------------------------------

            if arduino is not None:

                command = (
                    f"{label.upper()},"
                    f"{confidence * 100:.1f}\n"
                )

                arduino.write(
                    command.encode()
                )

                print(
                    "Sent to Arduino:",
                    command.strip()
                )


            apple_processed = True

            last_processed_time = current_time

            stable_count = 0

            previous_prediction = None


    # --------------------------------------------------------
    # RESET AFTER COOLDOWN
    # --------------------------------------------------------

    if apple_processed:

        if time.time() - last_processed_time > COOLDOWN_TIME:

            apple_processed = False


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

if arduino is not None:

    arduino.close()

print("System stopped.")