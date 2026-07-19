import cv2
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException

from api.services import (
    process_image_file, load_database, yolo_model, facenet_model, anti_spoof,
    device, LIVENESS_THRESHOLD, MATCH_TOLERANCE
)

router = APIRouter()

@router.post("/verify")
async def verify_identity(file: UploadFile = File(..., description="Image to verify for identity and liveness")):
    """
    Verify identity and check for liveness spoofing in the uploaded image.
    """
    frame = await process_image_file(file)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    known_encodings, known_names = load_database()
    if not known_encodings:
        raise HTTPException(status_code=500, detail="Database is empty. Please enroll users first.")

    results = yolo_model(frame, verbose=False)
    faces_detected = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            location = (y1, x2, y2, x1) # top, right, bottom, left
            
            # --- PER-FACE LIVENESS ---
            liveness_score = float(anti_spoof.get_liveness_score(frame, location))
            
            name = "Unknown"
            status = "Spoof"
            
            if liveness_score >= LIVENESS_THRESHOLD:
                # Identity Verification
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size != 0:
                    face_crop_resized = cv2.resize(face_crop, (160, 160))
                    face_crop_normalized = (face_crop_resized.astype(np.float32) - 127.5) / 128.0
                    face_tensor = torch.tensor(np.transpose(face_crop_normalized, (2, 0, 1))).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        encoding = facenet_model(face_tensor).cpu().numpy()[0]
                        dist = np.linalg.norm(known_encodings - encoding, axis=1)
                        if len(dist) > 0 and np.min(dist) < MATCH_TOLERANCE:
                            name = known_names[np.argmin(dist)]
                            status = "Verified"
                        else:
                            status = "Unregistered"
                            
            faces_detected.append({
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "liveness_score": round(liveness_score, 4),
                "status": status,
                "identity": name
            })

    if not faces_detected:
        return {"message": "No faces detected in the image", "faces": []}

    return {"message": "Verification complete", "faces": faces_detected}
