"""
@author: Sarshij K
"""
import cv2, os
import numpy as np
from PIL import Image

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    Ids = []

    for imagePath in imagePaths:
        # Convert image to grayscale
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')

        try:
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        except Exception as e:
            print(f"Skipping file {imagePath}: {e}")

    return faces, Ids

def TrainImages():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("❌ ERROR: cv2.face module not found. Please install opencv-contrib-python.")
        print("➡ Run this command: pip install opencv-contrib-python")
        return

    harcascadePath = "haarcascade_frontalface_default.xml"
    if not os.path.exists(harcascadePath):
        print(f"❌ ERROR: Haar Cascade file '{harcascadePath}' not found.")
        return

    detector = cv2.CascadeClassifier(harcascadePath)

    # Load training data
    faces, Ids = getImagesAndLabels("TrainingImage")

    if not faces:
        print("❌ ERROR: No training images found in 'TrainingImage/' folder.")
        return

    recognizer.train(faces, np.array(Ids))

    # Ensure TrainData directory exists
    os.makedirs("TrainData", exist_ok=True)

    # Save trained model
    recognizer.save(r"TrainData\Trainner.yml")
    print("✅ Image Trained and data stored in TrainData\\Trainner.yml")

TrainImages()

# python Train.py