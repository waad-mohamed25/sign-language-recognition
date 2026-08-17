#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe.python.solutions.hands as mp_hands
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# Project Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR / "dataset"

# =========================
# Get Classes
# =========================

classes = []

for item in os.listdir(dataset_path):
    full_path = os.path.join(dataset_path, item)

    if os.path.isdir(full_path):
        classes.append(item)

classes = sorted(classes)

print(len(classes))
print(classes)

# =========================
# MediaPipe Hand Detection
# =========================

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

data = []
labels = []

# =========================
# Feature Extraction
# =========================

for label in classes:

    folder = os.path.join(dataset_path, label)

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            x_list = []
            y_list = []

            for lm in hand.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)

            # Translation Normalization
            min_x = min(x_list)
            min_y = min(y_list)

            # Scale Normalization
            max_x = max(x_list)
            max_y = max(y_list)

            width = max_x - min_x
            height = max_y - min_y

            if width == 0 or height == 0:
                continue

            features = []

            for lm in hand.landmark:

                x = (lm.x - min_x) / width
                y = (lm.y - min_y) / height

                features.append(x)
                features.append(y)

            data.append(features)
            labels.append(str(label))

print(len(data))
print(len(labels))

# =========================
# Create DataFrame
# =========================

df = pd.DataFrame(data)

df["label"] = labels

print(df.shape)

df.head()

df["label"].value_counts()

for item in sorted(os.listdir(dataset_path)):
    print(item)

# =========================
# Save Dataset
# =========================

df.to_csv(
    BASE_DIR / "hand_landmarks_dataset_v2.csv",
    index=False
)

print("Dataset Saved")

# =========================
# X and y
# =========================

X = df.drop("label", axis=1)
y = df["label"]

print(X.shape)
print(y.shape)

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

# =========================
# Random Forest
# =========================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

print("Training Finished")

# =========================
# Prediction
# =========================

y_pred = rf.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

# =========================
# Evaluation
# =========================

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(16, 12))

sns.heatmap(
    cm,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

# =========================
# Save Model
# =========================

joblib.dump(
    rf,
    BASE_DIR / "sign_language_model_v2.pkl"
)

print("Model Saved Successfully")

# =========================
# Save Labels
# =========================

joblib.dump(
    sorted(y.unique()),
    BASE_DIR / "label_encoder_v2.pkl"
)

print("Labels Saved")

print(type(rf))
print(sorted(y.unique())[:5])