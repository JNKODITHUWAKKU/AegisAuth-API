import os
import cv2
import numpy as np
import torch

from ultralytics import YOLO

class LivenessDetection:
    def __init__(self, model_path="models/liveness_engine/weights/best.pt"):
        self.model_path = model_path
        
        if not os.path.exists(self.model_path):
            print(f"[WARNING] Custom liveness weights not found at {self.model_path}.")
            
        self.model = YOLO(self.model_path)

    def get_liveness_score(self, frame, face_location):
        top, right, bottom, left = face_location
        face_center_x = (left + right) / 2
        face_center_y = (top + bottom) / 2
        
        # Run inference on the FULL frame instead of a crop
        # YOLO models are trained on full images and need context
        results = self.model(frame, verbose=False)
        
        highest_real_score = 0.0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Check if this liveness bounding box overlaps with our detected face center
                if x1 < face_center_x < x2 and y1 < face_center_y < y2:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = r.names[cls_id].lower()
                    
                    # Check for authentic human presence
                    if class_name in ['real', 'live'] and conf > highest_real_score:
                        highest_real_score = conf

        return highest_real_score

    def reset_state(self):
        # State tracking is unneeded for dynamic YOLO frame-by-frame evaluation
        pass