from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
DETECTED_FOLDER = "static/detected"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DETECTED_FOLDER"] = DETECTED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

# Load YOLO v11 model
model = YOLO("best.pt")  # Ensure this file exists

def detect_objects(image_path):
    """Runs YOLO detection and draws bounding boxes."""
    results = model(image_path)
    image = cv2.imread(image_path)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}%"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Save the detected image
    detected_image_path = os.path.join(app.config["DETECTED_FOLDER"], os.path.basename(image_path))
    cv2.imwrite(detected_image_path, image)
    
    return detected_image_path

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected file"})

    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(image_path)

    detected_image_path = detect_objects(image_path)
    return jsonify({"image_url": detected_image_path})

@app.route("/static/detected/<filename>")
def detected_image(filename):
    return send_from_directory(app.config["DETECTED_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
