import numpy as np
import cv2
import torch
from ultralytics import YOLO
from flask import Flask, request, jsonify
from PIL import Image
import io
import gdown
import os

# Google Drive file ID (Replace with your actual File ID)
FILE_ID = "1KnIQI7bdwbHtPfCRqm-UooxyITLmGwEj"
OUTPUT_PATH = "crop_disease_model.pt"

# Check if model exists (to prevent re-downloading)
if not os.path.exists(OUTPUT_PATH):
    print("🔄 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", OUTPUT_PATH, quiet=False)
    print("✅ Model downloaded successfully!")
else:
    print("✅ Model already exists. Skipping download.")

# Load the YOLO model after downloading
from ultralytics import YOLO
model = YOLO(OUTPUT_PATH)

app = Flask(__name__)

# 🌿 Load YOLO Model
model = YOLO("crop_disease_model.pt")

# 🌱 Disease Dictionary
disease_info = {
    "bacterial spot": {
        "description": "Causes dark, water-soaked spots on leaves and fruits.",
        "remedy": "Remove infected plant debris, use copper-based fungicides.",
        "fertilizer": "Use potassium-rich fertilizers to strengthen plant immunity."
    },
    "early blight": {
        "description": "Causes dark concentric rings on leaves, leading to defoliation.",
        "remedy": "Apply fungicides, practice crop rotation.",
        "fertilizer": "Use nitrogen-rich fertilizers to improve plant resistance."
    },
    "healthy": {
        "description": "No signs of disease detected.",
        "remedy": "No action needed.",
        "fertilizer": "Balanced NPK fertilizers to maintain health."
    },
    "late blight": {
        "description": "Rapidly spreading fungal disease, common in wet conditions.",
        "remedy": "Remove infected plants, apply fungicides.",
        "fertilizer": "Calcium-based fertilizers reduce susceptibility."
    },
    "leaf miner": {
        "description": "Larvae create winding tunnels in leaves.",
        "remedy": "Use insecticidal sprays, remove affected leaves.",
        "fertilizer": "Organic compost or vermicompost to strengthen plants."
    },
    "leaf mold": {
        "description": "Yellowing leaves with fuzzy mold on undersides.",
        "remedy": "Improve air circulation, apply fungicides.",
        "fertilizer": "Phosphorus-based fertilizers to enhance root health."
    },
    "mosaic virus": {
        "description": "Causes mottled, yellow-green appearance on leaves.",
        "remedy": "Remove infected plants, control aphids.",
        "fertilizer": "Balanced NPK fertilizers, avoid excess nitrogen."
    },
    "septoria": {
        "description": "Small, circular brown spots on lower leaves.",
        "remedy": "Remove infected leaves, use fungicides.",
        "fertilizer": "Sulfur-based fertilizers to reduce fungal spread."
    },
    "spider mites": {
        "description": "Causes stippling and webbing on leaves.",
        "remedy": "Use miticides, introduce beneficial insects.",
        "fertilizer": "Silicon-based fertilizers improve resistance to pests."
    },
    "yellow leaf curl virus": {
        "description": "Curled, yellowing leaves and stunted growth.",
        "remedy": "Remove infected plants, control whiteflies.",
        "fertilizer": "Zinc and boron-based fertilizers to reduce symptoms."
    }
}

def transform_image(img):
    """Preprocess image for YOLO detection"""
    img = cv2.resize(img, (512, 512))  # Resize to match YOLO model
    return img

def detect_disease(img):
    """Run YOLO model and return disease predictions"""
    results = model(img)
    detections = results[0].boxes.data.tolist()
    classes = [model.names[int(detection[5])] for detection in detections]

    # Extract disease details
    table_data = []
    for cls in set(classes):
        details = disease_info.get(cls.lower(), {})
        table_data.append({
            "disease": cls,
            "description": details.get("description", "No description available"),
            "remedy": details.get("remedy", "No remedy available"),
            "fertilizer": details.get("fertilizer", "No recommendation available")
        })

    return table_data

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and returns disease diagnosis"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read()))
    img = np.array(img)
    
    processed_img = transform_image(img)
    result = detect_disease(processed_img)

    return jsonify({"diseases": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
