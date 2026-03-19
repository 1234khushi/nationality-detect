import cv2
import numpy as np
import joblib
from sklearn.cluster import KMeans
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from deepface import DeepFace

# ------------------ LOAD MODELS ------------------

# Load ML model and scaler
clf = joblib.load("/Users/khushimac/Downloads/nationality-detection/nationality_model.pkl")
scaler = joblib.load("/Users/khushimac/Downloads/nationality-detection/scaler.pkl")

# Load MobileNet (feature extractor)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)

# ------------------ NATIONALITY ------------------

def predict_nationality(img):

    img_resized = cv2.resize(img, (224, 224))
    img_processed = preprocess_input(img_resized)
    img_processed = np.expand_dims(img_processed, axis=0)

    feat = base_model.predict(img_processed)
    feat = feat.flatten().reshape(1, -1)

    feat = scaler.transform(feat)

    pred = int(clf.predict(feat)[0])

    label_map = {
        0: "Indian",
        1: "African",
        2: "US",
        3: "Other"
    }

    return label_map[pred]

# ------------------ DRESS COLOR ------------------

def detect_color(img):

    h, w, _ = img.shape

    # Focus on lower half (clothing)
    img = img[h//2:, :]

    img = cv2.resize(img, (100, 100))
    pixels = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]

    return dominant_color


def color_name(rgb):
    r, g, b = rgb

    if r > 150 and g < 100 and b < 100:
        return "Red"
    elif b > 150 and r < 100:
        return "Blue"
    elif g > 150 and r < 100:
        return "Green"
    elif r > 150 and g > 150:
        return "Yellow"
    elif r < 80 and g < 80 and b < 80:
        return "Black"
    elif r > 200 and g > 200 and b > 200:
        return "White"
    else:
        return "Mixed/Dark"

# ------------------ FINAL PIPELINE ------------------

def final_prediction(image):

    nationality = predict_nationality(image)

    analysis = DeepFace.analyze(
        image,
        actions=['emotion', 'age'],
        enforce_detection=False
    )

    emotion = analysis[0]['dominant_emotion']
    age = analysis[0]['age']

    result = {
        "Nationality": nationality,
        "Emotion": emotion
    }

    if nationality == "Indian":
        result["Age"] = age
        color_rgb = detect_color(image)
        result["Dress Color"] = color_name(color_rgb)

    elif nationality == "US":
        result["Age"] = age

    elif nationality == "African":
        color_rgb = detect_color(image)
        result["Dress Color"] = color_name(color_rgb)

    return result