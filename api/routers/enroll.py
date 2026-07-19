import os
import time
import pickle
import cv2
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List

from api.services import (
    process_image_file, yolo_model, facenet_model, 
    device, DATABASE_DIR
)

router = APIRouter()

@router.post("/enroll")
async def enroll_student(
    student_id: str = Form(..., description="Unique ID for the student"), 
    files: List[UploadFile] = File(..., description="List of face images (e.g., from different angles)")
):
    """
    Enroll a student by providing multiple images to extract identity anchors.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for enrollment")

    identity_embeddings = []
    
    for file in files:
        frame = await process_image_file(file)
        if frame is None: continue
            
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1], dtype=np.uint8)
        
        # YOLO Face detection
        results = yolo_model(rgb_frame, verbose=False)
        face_locations = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_locations.append((y1, x2, y2, x1)) # top, right, bottom, left
                
        if face_locations:
            # Take highest confidence face (first box)
            top, right, bottom, left = face_locations[0]
            top, bottom = max(0, top), min(frame.shape[0], bottom)
            left, right = max(0, left), min(frame.shape[1], right)
            face_crop = rgb_frame[top:bottom, left:right]
            
            if face_crop.size != 0:
                face_crop_resized = cv2.resize(face_crop, (160, 160))
                face_crop_normalized = (face_crop_resized.astype(np.float32) - 127.5) / 128.0
                face_tensor = torch.tensor(np.transpose(face_crop_normalized, (2, 0, 1))).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    encoding = facenet_model(face_tensor).cpu().numpy()[0]
                
                identity_embeddings.append(encoding)

    if not identity_embeddings:
        raise HTTPException(status_code=400, detail="No faces detected in the provided images.")

    # Save Profile
    target_file = os.path.join(DATABASE_DIR, f"auth_profile_{student_id}.pkl")
    
    # Append to existing vectors if profile already exists
    if os.path.exists(target_file):
        with open(target_file, "rb") as f:
            existing_payload = pickle.load(f)
            identity_embeddings = existing_payload.get("identity_anchors", []) + identity_embeddings

    profile_payload = {
        "metadata": {
            "student_id": student_id,
            "timestamp": time.time(),
            "metrics_captured": len(identity_embeddings)
        },
        "identity_anchors": identity_embeddings
    }
    
    with open(target_file, "wb") as storage_node:
        pickle.dump(profile_payload, storage_node)
        
    return {
        "message": "Enrollment successful",
        "student_id": student_id,
        "anchors_saved": len(identity_embeddings)
    }
