import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib

# Define the path to the dataset
DATASET_PATH = "TESS"
MODELS_PATH = "saved_models"  # Path to save the trained models

# Create the directory to save models if it doesn't exist
os.makedirs(MODELS_PATH, exist_ok=True)

# Emotions in the TESS dataset
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def load_data(dataset_path):
    features = []
    labels = []
    
    for emotion in EMOTIONS:
        emotion_path = os.path.join(dataset_path, f"OAF_{emotion}")
        for file in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file)
            if file_path.endswith('.wav'):
                mfccs = extract_features(file_path)
                features.append(mfccs)
                labels.append(emotion)
                
        emotion_path = os.path.join(dataset_path, f"YAF_{emotion}")
        for file in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file)
            if file_path.endswith('.wav'):
                mfccs = extract_features(file_path)
                features.append(mfccs)
                labels.append(emotion)
                
    return np.array(features), np.array(labels)

# Load the data
X, y = load_data(DATASET_PATH)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize different classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score for {name}: {accuracy_score(y_test, y_pred)}\n")
    
    # Save the trained model
    model_filename = os.path.join(MODELS_PATH, f"{name.replace(' ', '_')}_model.pkl")
    joblib.dump(clf, model_filename)
    print(f"Model saved as {model_filename}\n")

# Plot feature importances for Random Forest
importances = classifiers["Random Forest"].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
