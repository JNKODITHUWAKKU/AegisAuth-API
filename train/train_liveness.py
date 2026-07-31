import os
from ultralytics import YOLO

def train_anti_overfit_model():
    print("\n[SYSTEM] Initializing Liveness Engine with Anti-Overfitting Protocols...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(script_dir), 'models')
    base_model_path = os.path.join(models_dir, 'yolov8s.pt')
    
    model = YOLO(base_model_path)
    
    data_path = "/anti-spoofing-dataset"
    
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
        'project': os.path.join(os.path.dirname(script_dir), 'models'),
        'name': 'liveness_engine'
    }

    model.train(**training_args)
    print("\n[SYSTEM] Optimal weights saved to models/liveness_engine/weights/best.pt")

if __name__ == '__main__':
    train_anti_overfit_model()