import os
import mlflow
from ultralytics import YOLO

def train_anti_overfit_model():
    print("\n[SYSTEM] Initializing Liveness Engine with Anti-Overfitting Protocols...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(script_dir), 'models')
    base_model_path = os.path.join(models_dir, 'yolov8s.pt')
    
    model = YOLO(base_model_path)
    
    data_path = "./anti-spoofing-dataset"
    
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
        'project': os.path.join(os.path.dirname(script_dir), 'runs'),
        'name': 'liveness_engine',
        'exist_ok': True
    }

    # Setup MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Liveness_Detection_Experiment")
    
    with mlflow.start_run(run_name="YOLOv8s_Anti_Spoofing"):
        # Log parameters
        mlflow.log_params({
            k: v for k, v in training_args.items() if k not in ['data', 'project', 'name']
        })
        
        # Train model
        results = model.train(**training_args)
        
        # Log final model weights as MLflow artifact
        best_model_path = os.path.join(training_args['project'], training_args['name'], 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path="model_weights")
            
            # --- MODEL REGISTRY ---
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model_weights"
            
            # Register the model
            model_details = mlflow.register_model(model_uri=model_uri, name="YOLOv8_Liveness_Engine")
            print(f"\n[SYSTEM] Model registered in MLflow Model Registry as Version {model_details.version}")
            
        # Log final metrics if available
        if hasattr(results, 'results_dict'):
            mlflow.log_metrics({str(k): float(v) for k, v in results.results_dict.items()})

    print("\n[SYSTEM] Optimal weights saved and logged to MLflow")

if __name__ == '__main__':
    train_anti_overfit_model()