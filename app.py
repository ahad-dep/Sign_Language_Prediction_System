from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "model/sign_language_model.keras"
CLASS_NAMES_PATH = "model/class_names.txt"

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]

print("AI model loaded successfully!")
print("Classes:", class_names)


# Latest prediction
latest_prediction = "--"
latest_confidence = 0.0


# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# --------------------------------------------------
# WEBCAM
# --------------------------------------------------

print("Opening webcam...")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
else:
    print("Webcam opened successfully!")


# --------------------------------------------------
# GENERATE VIDEO FRAMES
# --------------------------------------------------

def generate_frames():

    global latest_prediction
    global latest_confidence

    while True:

        success, frame = camera.read()

        if not success:

            print("ERROR: Could not read webcam frame.")

            # Try reopening the camera
            camera.release()

            camera.open(0, cv2.CAP_DSHOW)

            continue


        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape


        # --------------------------------------------------
        # MEDIAPIPE HAND DETECTION
        # --------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)


        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )


                # --------------------------------------------------
                # FIND HAND BOUNDING BOX
                # --------------------------------------------------

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


                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255, 0, 255),
                    2
                )


                # --------------------------------------------------
                # CROP HAND
                # --------------------------------------------------

                hand_image = frame[
                    y_min:y_max,
                    x_min:x_max
                ]


                if hand_image.size == 0:
                    continue


                # Resize
                hand_image = cv2.resize(
                    hand_image,
                    (128, 128)
                )


                # BGR -> RGB
                hand_image = cv2.cvtColor(
                    hand_image,
                    cv2.COLOR_BGR2RGB
                )


                # Normalize
                hand_image = hand_image / 255.0


                # Add batch dimension
                hand_image = np.expand_dims(
                    hand_image,
                    axis=0
                )


                # --------------------------------------------------
                # AI PREDICTION
                # --------------------------------------------------

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


                # Update prediction
                latest_prediction = predicted_class
                latest_confidence = float(
                    confidence
                )


                # --------------------------------------------------
                # DISPLAY PREDICTION
                # --------------------------------------------------

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


        else:

            latest_prediction = "--"
            latest_confidence = 0.0

            cv2.putText(
                frame,
                "Show your hand",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        # --------------------------------------------------
        # CONVERT FRAME TO JPEG
        # --------------------------------------------------

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()


        # --------------------------------------------------
        # SEND FRAME TO BROWSER
        # --------------------------------------------------

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

@app.route("/prediction")
def prediction():

    return render_template(
        "prediction.html"
    )


# --------------------------------------------------
# VIDEO FEED
# --------------------------------------------------

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# PREDICTION DATA
# --------------------------------------------------

@app.route("/prediction_data")
def prediction_data():

    return jsonify({
        "prediction": latest_prediction,
        "confidence": round(
            latest_confidence,
            2
        )
    })


# --------------------------------------------------
# RUN FLASK
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )