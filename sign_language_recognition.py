#!/usr/bin/env python
# coding: utf-8

import cv2
import joblib
import numpy as np
import warnings

from pathlib import Path
from collections import deque, Counter

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

import pyttsx3

warnings.filterwarnings("ignore")

# =========================
# Project Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sign_language_model_v2.pkl"
LABELS_PATH = BASE_DIR / "label_encoder_v2.pkl"
WORDS_PATH = BASE_DIR / "words.txt"

# =========================
# Load Model & Labels
# =========================

model = joblib.load(MODEL_PATH)
labels = joblib.load(LABELS_PATH)

print("Model Loaded Successfully")
print(type(model))
print("Classes:", len(labels))
print(labels)

# =========================
# Text To Speech
# =========================

def speak(text):

    engine = pyttsx3.init()

    engine.say(text)

    engine.runAndWait()

    engine.stop()

# =========================
# Load Dictionary
# =========================

with open(WORDS_PATH, "r") as f:
    WORDS = [word.strip().upper() for word in f]

print(len(WORDS))

# =========================
# Dictionary Suggestion
# =========================

def get_suggestion(text):

    if text.endswith(" "):
        return ""

    if text.strip() == "":
        return ""

    last_word = text.split()[-1].upper()

    if len(last_word) < 3:
        return ""

    for word in WORDS:

        if word.startswith(last_word) and word != last_word:
            return word

    return ""

# =========================
# MediaPipe Hands
# =========================

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Prediction Smoothing
# =========================

predictions = deque(maxlen=20)

# =========================
# Sentence Building
# =========================

current_word = ""

last_letter = ""

stable_count = 0

ADD_THRESHOLD = 20

no_hand_frames = 0

# =========================
# Camera
# =========================

cap = cv2.VideoCapture(0)

# =========================
# Window Settings
# =========================

cv2.namedWindow(
    "Sign Language Recognition",
    cv2.WINDOW_NORMAL
)

# =========================
# Camera Loop
# =========================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        no_hand_frames = 0

        for handedness in results.multi_handedness:
            pass

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw Landmarks

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:

                x_list.append(lm.x)
                y_list.append(lm.y)

            min_x = min(x_list)
            min_y = min(y_list)

            max_x = max(x_list)
            max_y = max(y_list)

            width = max_x - min_x
            height = max_y - min_y

            if width > 0 and height > 0:

                features = []

                for lm in hand_landmarks.landmark:

                    # IMPORTANT:
                    # Keep the original hand orientation correction

                    if handedness.classification[0].label == "Right":

                        x = 1 - (
                            (lm.x - min_x) / width
                        )

                    else:

                        x = (
                            lm.x - min_x
                        ) / width

                    y = (
                        lm.y - min_y
                    ) / height

                    features.append(x)
                    features.append(y)

                features = np.array(
                    features
                ).reshape(1, -1)

                # =========================
                # Prediction
                # =========================

                probs = model.predict_proba(
                    features
                )[0]

                pred_index = np.argmax(probs)

                prediction = model.classes_[
                    pred_index
                ]

                confidence = probs[
                    pred_index
                ]

                # =========================
                # Stable Prediction
                # =========================

                if prediction == last_letter:

                    stable_count += 1

                else:

                    stable_count = 0

                    last_letter = prediction

                if stable_count == ADD_THRESHOLD:

                    current_word += prediction

                    stable_count = 0

                # =========================
                # Smoothing
                # =========================

                predictions.append(
                    prediction
                )

                final_prediction = Counter(
                    predictions
                ).most_common(1)[0][0]

                # =========================
                # Display Letter
                # =========================

                cv2.putText(
                    frame,
                    f"Letter: {prediction}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    else:

        no_hand_frames += 1

    # =========================
    # Display Word
    # =========================

    cv2.putText(
        frame,
        f"Word: {current_word}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # =========================
    # Dictionary Suggestion
    # =========================

    suggestion = get_suggestion(
        current_word
    )

    cv2.putText(
        frame,
        f"Suggestion: {suggestion}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # =========================
    # Show Camera
    # =========================

    cv2.imshow(
        "Sign Language Recognition",
        frame
    )

    # =========================
    # Keyboard Controls
    # =========================

    key = cv2.waitKey(1) & 0xFF

    # Full screen

    if key == ord('f'):

        cv2.setWindowProperty(
            "Sign Language Recognition",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    # Windowed screen

    if key == ord('w'):

        cv2.setWindowProperty(
            "Sign Language Recognition",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL
        )

    # Clear current word

    if key == ord('c'):

        current_word = ""

    # Space bar

    if key == ord(' '):

        current_word += "   "

    # Backspace

    if key == 8:

        if len(current_word) > 0:

            current_word = current_word[:-1]

    # Speak current word

    if key == ord('s'):

        if current_word.strip() == "":

            speak("No text available")

        else:

            speak(current_word)

    # Accept suggestion

    if key == 13:

        if suggestion != "":

            words = current_word.split()

            if len(words) > 0:

                words[-1] = suggestion

                current_word = (
                    "   ".join(words)
                    + "   "
                )

    # Escape

    if key == 27:

        break

# =========================
# Cleanup
# =========================

cap.release()

cv2.destroyAllWindows()