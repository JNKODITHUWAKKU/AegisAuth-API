import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient

# Get absolute path to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MLRUNS_DB_PATH = os.path.join(PROJECT_ROOT, "mlruns.db")

def fetch_production_model(model_name="YOLOv8_Liveness_Engine", dest_path=None):
    """
    Fetches the latest 'Production' (or fallback) version of the specified model from the MLflow Model Registry 
    and saves it to the local models directory for the FastAPI inference engine to use.
    """
    if dest_path is None:
        dest_path = os.path.join(PROJECT_ROOT, "models", "liveness_engine", "weights", "best.pt")

    print(f"\n[SYSTEM] Connecting to MLflow Model Registry (sqlite:///{MLRUNS_DB_PATH})...")
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DB_PATH}")
    client = MlflowClient()

    try:
        # Get the latest versions for the model
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            print(f"[WARNING] No model found in 'Production' stage for '{model_name}'.")
            print("Checking for 'None' or 'Staging' stages as fallback...")
            latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
            
            if not latest_versions:
                print(f"[ERROR] No versions found for model '{model_name}' in the registry.")
                return
        
        # Take the most recent version available in the target stages
        latest_versions.sort(key=lambda x: x.version, reverse=True)
        prod_model = latest_versions[0]
        
        print(f"[SYSTEM] Found {model_name} Version {prod_model.version} (Stage: {prod_model.current_stage})")
        
        # Download the artifact
        print(f"[SYSTEM] Downloading model artifacts...")
        # Note: DVC or git might be tracking the destination, so we download to a temp location first
        downloaded_path = client.download_artifacts(prod_model.run_id, "model_weights/best.pt", dst_path="/tmp")
        
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Safely remove the old model if it exists (avoids DVC read-only permission errors)
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        
        # Move the downloaded file to the production destination
        shutil.move(downloaded_path, dest_path)
        print(f"[SUCCESS] Production model successfully deployed to {dest_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch model from registry: {e}")

if __name__ == "__main__":
    fetch_production_model()
