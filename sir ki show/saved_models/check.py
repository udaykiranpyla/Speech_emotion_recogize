import os
import librosa
import numpy as np
import joblib

# Define the path to the saved models
MODELS_PATH = "saved_models"

# Emotions in the TESS dataset
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def predict_emotion(file_path, model_name):
    # Load the saved model
    model_filename = os.path.join(MODELS_PATH, f"{model_name.replace(' ', '_')}_model.pkl")
    model = joblib.load(model_filename)

    # Extract features from the input file
    features = extract_features(file_path).reshape(1, -1)

    # Predict the emotion
    prediction = model.predict(features)
    return prediction[0]

# Example usage:
if __name__ == "__main__":
    # Path to the input audio file
    input_file_path = "input.wav"
    
    # Choose the model you want to use for prediction
    model_name = "Random Forest"  # Change to the desired model name

    # Predict the emotion
    predicted_emotion = predict_emotion(input_file_path, model_name)
    
    print(f"The predicted emotion is: {predicted_emotion}")
