import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import soundfile as sf
import librosa
import numpy as np
from keras.models import load_model

# Load the trained emotion recognition model
emotion_model = load_model("C:\\emotional\\emotion_model.h5")

# Define the emotions and their corresponding emojis
emotions_emojis = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲"
}

# Function to recognize emotions from speech and update the GUI
def recognize_emotion():
    # Open a file dialog to select an audio file
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    
    if file_path:
        # Read the audio file and extract features
        data, sample_rate = librosa.load(file_path, duration=2.4, offset=0.6)
        features = get_features(data, sample_rate)
        
        if len(features) > 0:
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=3)
            features = np.swapaxes(features, 1, 2)
            
            # Predict the emotion using the trained model
            prediction = emotion_model.predict(features)
            emotion_index = np.argmax(prediction)
            predicted_emotion = get_emotion(emotion_index + 1)
            
            # Update the GUI with the predicted emotion and emoji
            emotion_label.configure(text=predicted_emotion)
            emoji_label.configure(text=emotions_emojis[predicted_emotion])

# Function to extract features from audio data
def get_features(data, sr):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return mfcc

# Function to get the emotion label based on the index
def get_emotion(number):
    info = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
    return info[number]

# Create the GUI window
window = tk.Tk()
window.title("Speech Emotion Recognition")
window.geometry("300x300")

# Create GUI elements
title_label = tk.Label(window, text="Speech Emotion Recognition", font=("Arial", 14))
title_label.pack(pady=20)

button = tk.Button(window, text="Select Audio File", command=recognize_emotion)
button.pack(pady=10)

emotion_label = tk.Label(window, text="", font=("Arial", 20))
emotion_label.pack(pady=10)

emoji_label = tk.Label(window, text="", font=("Arial", 50))
emoji_label.pack(pady=10)

# Start the GUI event loop
window.mainloop()
