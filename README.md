Overview

This project focuses on building a Speech Emotion Recognition (SER) System that classifies human emotions such as happiness, sadness, anger, and neutrality based on spoken language. The goal is to develop a robust machine-learning model that enhances human-computer interaction by accurately identifying emotions from speech signals.

🏆 Problem Statement

Understanding human emotions is crucial for intelligent systems that interact effectively with users. This project aims to classify the emotional state of a speaker from audio recordings using machine learning and deep learning techniques.

⚡ Challenges:

Variations in accents and speaking speeds.

Background noise affecting recognition.

Overlapping acoustic features making differentiation difficult.

🔬 Solution:

Use Mel-Frequency Cepstral Coefficients (MFCCs) for feature extraction.

Train models using Convolutional Neural Networks (CNNs), Support Vector Machines (SVMs), and Random Forest classifiers.

Evaluate using accuracy, precision, recall, and F1-score.

Provide a Graphical User Interface (GUI) using Tkinter for real-time predictions.

🚀 Features

Emotion classification into Angry, Happy, Sad, and Neutral.

Uses MFCCs for feature extraction.

Machine learning models: CNN, SVM, and Random Forest.

GUI for easy interaction.

Applications in call centers, mental health monitoring, and virtual assistants.

🛠️ Technologies Used

Python 3.6+

TensorFlow – Deep Learning framework

Librosa – Audio processing and feature extraction

NumPy – Numerical computations

Scikit-Learn – Machine learning utilities

Joblib – Model saving/loading

Tkinter – GUI development

📊 Dataset Used

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

TESS (Toronto Emotional Speech Set)

CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)

SAVEE (Surrey Audio-Visual Expressed Emotion)

📸 GUI Preview

Below is an example of the GUI interface for real-time speech emotion recognition:![IMG_6840](https://github.com/user-attachments/assets/6c07a168-b773-457d-a76f-cb43ed76da44)
![IMG_6838](https://github.com/user-attachments/assets/aaa12f8a-c2ff-45e2-8161-f0fb807abe55)



📝 Sample Predictions

Example output from the system when analyzing a speech file:

Loading audio file...
Extracting features...
Predicted Emotion: Happy
Confidence Score: 92%

📦 Installation & Setup

Clone the repository:

git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

🎯 Usage

Run the GUI and select an audio file to classify emotions.

The system will analyze the speech and display the predicted emotion.

Use the model in real-time applications for sentiment analysis.

🏆 Future Improvements

Expand the dataset for better generalization.

Improve accuracy with transformer-based models.

Integrate real-time microphone input for instant analysis.

🤝 Contributions

Feel free to contribute by submitting pull requests or reporting issues.

📜 License

This project is licensed under the MIT License.
