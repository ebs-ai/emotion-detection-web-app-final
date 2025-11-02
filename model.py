import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Confirm working directory
print("📂 Current working directory:", os.getcwd())

# Change directory if needed
# os.chdir("C:/Users/HomePC/PycharmProjects/emotiondetectionproject")

# Load your fine-tuned model
MODEL_PATH = "emotion_model_final.h5"
assert os.path.exists(MODEL_PATH), f"❌ Model file not found at {MODEL_PATH}"

model = load_model(MODEL_PATH)
print("✅ Emotion model loaded successfully!")

# Emotion labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def get_emotion(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img / 255.0
        img = np.expand_dims(img, axis=(0, -1))
        preds = model.predict(img)
        emotion = class_labels[np.argmax(preds)]
        return emotion
    except Exception as e:
        return f"Error: {str(e)}"

