import os
import pickle
import cv2
import numpy as np
import torch
from fastapi import UploadFile

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from api.liveness import LivenessDetection

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.getenv("AEGISAUTH_DB_DIR", os.path.join(PROJECT_ROOT, "database", "identities"))
os.makedirs(DATABASE_DIR, exist_ok=True)
MATCH_TOLERANCE = 0.85 
LIVENESS_THRESHOLD = 0.65

# Initialize Global Models (Loaded once on startup)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8n-face.pt")
LIVENESS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "liveness_engine", "weights", "best.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
anti_spoof = LivenessDetection(model_path=LIVENESS_MODEL_PATH)

def load_database():
    known_encodings, known_names = [], []
    if not os.path.exists(DATABASE_DIR): 
        return [], []
    for filename in os.listdir(DATABASE_DIR):
        if filename.endswith(".pkl"):
            with open(os.path.join(DATABASE_DIR, filename), "rb") as file:
                payload = pickle.load(file)
                for encoding in payload["identity_anchors"]:
                    known_encodings.append(encoding)
                    known_names.append(payload["metadata"]["student_id"])
    return known_encodings, known_names

async def process_image_file(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame
