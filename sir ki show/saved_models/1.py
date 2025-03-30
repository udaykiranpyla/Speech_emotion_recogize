import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import librosa
import numpy as np
import joblib

# Define the path to the saved models
MODELS_PATH = "saved_models"

# Emotions in the TESS dataset
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

# Extract features from audio file
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Predict emotion from audio file
def predict_emotion(file_path, model_name):
    try:
        model_filename = os.path.join(MODELS_PATH, f"{model_name.replace(' ', '_')}_model.pkl")
        model = joblib.load(model_filename)
        features = extract_features(file_path).reshape(1, -1)
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        return None

# GUI Application
class EmotionRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Speech Emotion Recognition")
        self.geometry("600x300")
        
        self.label = tk.Label(self, text="Choose an audio file and a model to predict the emotion", wraplength=300)
        self.label.pack(pady=10)
        
        self.file_path = ""
        
        self.choose_button = tk.Button(self, text="Choose Audio File", command=self.choose_file)
        self.choose_button.pack(pady=5)
        
        self.model_label = tk.Label(self, text="Select Model")
        self.model_label.pack(pady=5)
        
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self, textvariable=self.model_var)
        self.model_combobox['values'] = ["Random Forest", "SVM", "k-NN", "Logistic Regression", "Gradient Boosting", "MLP Classifier"]
        self.model_combobox.current(0)
        self.model_combobox.pack(pady=5)
        
        self.predict_button = tk.Button(self, text="Predict Emotion", command=self.predict_emotion)
        self.predict_button.pack(pady=5)
        
        self.result_label = tk.Label(self, text="", wraplength=300)
        self.result_label.pack(pady=10)
        
    def choose_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if self.file_path:
            self.label.config(text=f"Selected File: {os.path.basename(self.file_path)}")
    
    def predict_emotion(self):
        if not self.file_path:
            messagebox.showwarning("Warning", "Please choose an audio file first!")
            return
        
        model_name = self.model_var.get()
        predicted_emotion = predict_emotion(self.file_path, model_name)
        
        if predicted_emotion:
            self.result_label.config(text=f"The predicted emotion is: {predicted_emotion}")
        else:
            self.result_label.config(text="")

if __name__ == "__main__":
    app = EmotionRecognitionApp()
    app.mainloop()
