from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)  # Return number of faces detected

@app.route("/detect_faces", methods=["POST"])
def detect():
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    image = np.array(image)
    face_count = detect_faces(image)

    return jsonify({"faces_detected": face_count})

if __name__ == "__main__":
    app.run(debug=True)
