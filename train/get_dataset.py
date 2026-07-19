import os
from roboflow import Roboflow

def download_dataset():
    print("[SYSTEM] Fetching Enterprise Face Anti-Spoofing Dataset...")
    rf = Roboflow(api_key="")
    project = rf.workspace("face-anti-spoofing-detection").project("face-anti-spoofing-detection")
    version = project.version(2)
    
    # Force download to local fast storage on Colab
    dataset_path = "/content/face-anti-spoofing-detection-2" if os.path.exists("/content") else "./face-anti-spoofing-detection-2"
    
    dataset = version.download("yolov8", location=dataset_path)
    
    yaml_path = f"{dataset.location}/data.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            yaml_data = file.read()
        
        # Fix the relative paths
        yaml_data = yaml_data.replace('../train/images', 'train/images')
        yaml_data = yaml_data.replace('../valid/images', 'valid/images')
        yaml_data = yaml_data.replace('../test/images', 'test/images')
        
        with open(yaml_path, 'w') as file:
            file.write(yaml_data)
        print("[SYSTEM] Fixed corrupted pathing in data.yaml")

    print(f"[SYSTEM] Dataset secured at: {dataset.location}")

if __name__ == '__main__':
    download_dataset()
