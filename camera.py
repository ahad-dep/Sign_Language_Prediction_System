import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

MODEL_PATH = "model/sign_language_model.keras"
CLASS_NAMES_PATH = "model/class_names.txt"

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("AI model loaded successfully!")


# ==================================================
# LOAD CLASS NAMES
# ==================================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]

print("Classes loaded:")
print(class_names)


# ==================================================
# START WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam.")
    exit()


# ==================================================
# MEDIAPIPE HAND DETECTION
# ==================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Get frame size
    height, width, _ = frame.shape


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect hand
    results = hands.process(rgb_frame)


    # ==================================================
    # IF HAND IS DETECTED
    # ==================================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:


            # ------------------------------------------
            # DRAW HAND LANDMARKS
            # ------------------------------------------

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            # ------------------------------------------
            # FIND HAND BOUNDING BOX
            # ------------------------------------------

            x_coordinates = [
                int(landmark.x * width)
                for landmark in hand_landmarks.landmark
            ]

            y_coordinates = [
                int(landmark.y * height)
                for landmark in hand_landmarks.landmark
            ]


            x_min = max(
                min(x_coordinates) - 30,
                0
            )

            x_max = min(
                max(x_coordinates) + 30,
                width
            )

            y_min = max(
                min(y_coordinates) - 30,
                0
            )

            y_max = min(
                max(y_coordinates) + 30,
                height
            )


            # ------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (255, 0, 255),
                2
            )


            # ------------------------------------------
            # CROP HAND
            # ------------------------------------------

            hand_image = frame[
                y_min:y_max,
                x_min:x_max
            ]


            if hand_image.size == 0:
                continue


            # ------------------------------------------
            # PREPARE IMAGE FOR CNN
            # ------------------------------------------

            hand_image = cv2.resize(
                hand_image,
                (128, 128)
            )

            hand_image = cv2.cvtColor(
                hand_image,
                cv2.COLOR_BGR2RGB
            )

            hand_image = hand_image / 255.0

            hand_image = np.expand_dims(
                hand_image,
                axis=0
            )


            # ------------------------------------------
            # AI PREDICTION
            # ------------------------------------------

            predictions = model.predict(
                hand_image,
                verbose=0
            )


            predicted_index = np.argmax(
                predictions[0]
            )

            confidence = (
                predictions[0][predicted_index]
                * 100
            )


            predicted_class = class_names[
                predicted_index
            ]


            # ------------------------------------------
            # DISPLAY PREDICTION
            # ------------------------------------------

            cv2.putText(
                frame,
                f"Prediction: {predicted_class}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


    # ==================================================
    # NO HAND DETECTED
    # ==================================================

    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ==================================================
    # DISPLAY CAMERA
    # ==================================================

    cv2.imshow(
        "SignAI - Sign Language Prediction",
        frame
    )


    # ==================================================
    # PRESS Q TO EXIT
    # ==================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==================================================
# RELEASE RESOURCES
# ==================================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("Camera closed.")