import google.generativeai as genai
import os
import io
from flask import Flask, request, jsonify
from PIL import Image

# 🌱 Load Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found. Set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and returns disease diagnosis using Gemini"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read()))

    # Convert image to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # 🔍 Send image to Gemini Vision API
    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_content([img_bytes])

    # Extract response
    result = response.text.strip() if response.text else "No diagnosis available"
    
    return jsonify({"disease_diagnosis": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
