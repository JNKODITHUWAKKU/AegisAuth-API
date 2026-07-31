import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

def download_dataset():
    print("[SYSTEM] Fetching Enterprise Face Anti-Spoofing Dataset...")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY environment variable not set. Please set it in your .env file.")
        print("[INFO] Skipping dataset download. Ensure dataset exists in ./anti-spoofing-dataset")
        return
        
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("face-anti-spoofing-detection").project("face-anti-spoofing-detection")
    version = project.version(2)
    
    dataset_path = "./anti-spoofing-dataset"
    
    dataset = version.download("yolov8", location=dataset_path)
    
    yaml_path = f"{dataset.location}/data.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            yaml_data = file.read()
        
        yaml_data = yaml_data.replace('../train/images', 'train/images')
        yaml_data = yaml_data.replace('../valid/images', 'valid/images')
        yaml_data = yaml_data.replace('../test/images', 'test/images')
        
        with open(yaml_path, 'w') as file:
            file.write(yaml_data)
        print("[SYSTEM] Fixed corrupted pathing in data.yaml")

    print(f"[SYSTEM] Dataset secured at: {dataset.location}")

if __name__ == '__main__':
    download_dataset()
