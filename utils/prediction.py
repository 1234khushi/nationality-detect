import numpy as np
import joblib
from sklearn.cluster import KMeans
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from PIL import Image
from pathlib import Path

# ------------------ LOAD MODELS ------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "nationality_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

clf = None
scaler = None
base_model = None


def get_classifier():
    global clf
    if clf is None:
        clf = joblib.load(MODEL_PATH)
    return clf


def get_scaler():
    global scaler
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
    return scaler


def get_feature_extractor():
    global base_model
    if base_model is None:
        try:
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'
            )
        except Exception as exc:
            raise RuntimeError(
                "MobileNetV2 weights could not be loaded. Ensure network access is available on first run or pre-cache the TensorFlow weights."
            ) from exc
    return base_model

# ------------------ NATIONALITY ------------------

def predict_nationality(img):
    img_resized = np.array(Image.fromarray(img).resize((224, 224)))
    img_processed = preprocess_input(img_resized)
    img_processed = np.expand_dims(img_processed, axis=0)

    feat = get_feature_extractor().predict(img_processed)
    feat = feat.flatten().reshape(1, -1)

    feat = get_scaler().transform(feat)

    pred = int(get_classifier().predict(feat)[0])

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

    img = np.array(Image.fromarray(img).resize((100, 100)))
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


def analyze_face(image):
    try:
        from deepface import DeepFace
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'deepface'. Install project requirements before running predictions."
        ) from exc

    return DeepFace.analyze(
        image,
        actions=['emotion', 'age'],
        enforce_detection=False
    )

# ------------------ FINAL PIPELINE ------------------

def final_prediction(image):

    nationality = predict_nationality(image)

    analysis = analyze_face(image)

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
