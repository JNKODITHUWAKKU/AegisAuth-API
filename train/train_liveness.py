import os
from ultralytics import YOLO

def train_anti_overfit_model():
    print("\n[SYSTEM] Initializing Liveness Engine with Anti-Overfitting Protocols...")
    model = YOLO('yolov8s.pt')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = "/content/face-anti-spoofing-detection-2" if os.path.exists("/content") else "./face-anti-spoofing-detection-2"
    
    training_args = {
        'data': f"{data_path}/data.yaml",
        'epochs': 50,             
        'patience': 7,            
        'batch': 32,               
        'optimizer': 'AdamW',      
        'lr0': 0.001,              
        'weight_decay': 0.0005,    
        'dropout': 0.2,            
        'mosaic': 1.0,             
        'erasing': 0.4,            
        'imgsz': 640,              
        'mixup': 0.0,              
        'project': os.path.join(script_dir, 'LecAttend_AI'),
        'name': 'liveness_engine'
    }

    model.train(**training_args)
    print("\n[SYSTEM] Optimal weights saved to LecAttend_AI/liveness_engine/weights/best.pt")

if __name__ == '__main__':
    train_anti_overfit_model()