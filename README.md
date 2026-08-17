# Sign Language Recognition System

A real-time sign language recognition system that uses computer vision and machine learning to recognize hand gestures through a webcam and convert them into letters.

## Project Overview

This project uses MediaPipe Hands to detect hand landmarks from camera frames and a Random Forest classifier to recognize sign language gestures.

The system supports:

- Digits 0–9
- English alphabet A–Z
- Real-time webcam recognition
- Prediction confidence
- Word building
- Word suggestions
- Text-to-speech
- Keyboard controls

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Random Forest
- Joblib
- Matplotlib
- Seaborn
- Pyttsx3

## Project Structure

```text
sign-language-recognition/
│
├── dataset/
│
├── sign_language_recognition.py
├── train_model.py
│
├── sign_language_model_v2.pkl
├── label_encoder_v2.pkl
├── hand_landmarks_dataset_v2.csv
├── words.txt
│
├── sign_language_recognition_analysis.ipynb
├── requirements.txt
└── README.md