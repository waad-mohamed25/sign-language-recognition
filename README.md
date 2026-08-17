# Sign Language Recognition System

A real-time sign language recognition system that uses computer vision and machine learning to recognize hand gestures through a webcam and convert them into letters.

# Project Overview

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

# Technologies Used

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

# Project Structure

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


## How It Works

The system follows these main steps:

1. The webcam captures a live video frame.
2. MediaPipe Hands detects the hand and its landmarks.
3. The hand landmark coordinates are normalized.
4. Hand orientation is handled before prediction.
5. The trained Random Forest model predicts the corresponding sign.
6. The predicted letter is displayed in real time.
7. Recognized letters can be combined into words.
8. The system provides word suggestions using the dictionary.
9. The recognized text can be converted to speech.

## Machine Learning Model

The project uses a Random Forest classifier trained on normalized hand landmark features.

The trained model is stored in:
sign_language_model_v2.pkl


The class labels are stored in:
label_encoder_v2.pkl


The extracted hand landmark dataset is stored in:
hand_landmarks_dataset_v2.csv


## Model Performance

The trained model achieved the following results on the test set:

- Accuracy: 96.98%
- Precision: 97.09%
- Recall: 96.98%
- F1 Score: 96.90%

## Installation

Install the required Python packages using:

pip install -r requirements.txt

Python 3.12 is recommended for this project.

## Running the Application

The trained model is already included in the project, so the recognition application can be run directly without retraining.

Run:

python sign_language_recognition.py


On Windows, if multiple Python versions are installed:

py -3.12 sign_language_recognition.py


The application will load the trained model and open the webcam.

## Training the Model

To retrain the model using the dataset, run:

python train_model.py


On Windows:
py -3.12 train_model.py


The training script:

1. Reads the image dataset.
2. Detects hand landmarks using MediaPipe.
3. Normalizes the landmark coordinates.
4. Creates the hand landmark dataset.
5. Splits the data into training and testing sets.
6. Trains the Random Forest classifier.
7. Evaluates the model.
8. Generates the confusion matrix.
9. Saves the trained model and labels.

## Keyboard Controls

| Key | Action |
|-----|--------|
| `F` | Full-screen mode |
| `W` | Windowed mode |
| `C` | Clear current text |
| `Space` | Add a space |
| `Backspace` | Delete the last character |
| `S` | Speak the current text |
| `Enter` | Accept a word suggestion |
| `Esc` | Exit the application |

## Notebook

The notebook:
sign_language_recognition_analysis.ipynb


contains the project analysis, data processing, model development, training, evaluation, and presentation material.

## Dataset

The original image dataset is kept locally and is excluded from Git using `.gitignore` because of its size.

The generated hand-landmark dataset is included in the repository as:

hand_landmarks_dataset_v2.csv


## Notes

The trained model is already included in the repository, so users do not need to retrain the model before running the application.

Retraining is only necessary when updating the dataset or experimenting with the model.

## Author

**Waad Mohamed**

## Project Information

**Instructor:** Eng. Mohamed Walied