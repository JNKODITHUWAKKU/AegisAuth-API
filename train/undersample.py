import os
import random
import shutil

def undersample():
    base_dir = "/content/face-anti-spoofing-detection-2" if os.path.exists("/content") else "./face-anti-spoofing-detection-2"
    print("\n[SYSTEM] Balancing dataset natively to remove majority class bias...")
    splits = ['train', 'valid', 'test']
    
    backup_dir = os.path.join(base_dir, 'removed_majority_samples')
    os.makedirs(backup_dir, exist_ok=True)
    total_removed = 0
    
    for split in splits:
        labels_dir = os.path.join(base_dir, split, 'labels')
        images_dir = os.path.join(base_dir, split, 'images')
        
        if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
            continue
            
        majority_images = []
        minority_count = 0
        
        for filename in os.listdir(labels_dir):
            if not filename.endswith('.txt'): continue
            filepath = os.path.join(labels_dir, filename)
            has_majority = False
            has_minority = False
            
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id == 0: has_majority = True
                        elif cls_id == 1: has_minority = True
                            
            if has_minority:
                minority_count += 1
            elif has_majority and not has_minority:
                majority_images.append(filename)
                
        num_to_remove = len(majority_images) - minority_count
        if num_to_remove > 0:
            print(f"[{split}] Undersampling {num_to_remove} majority (Fake) images...")
            images_to_remove = random.sample(majority_images, num_to_remove)
            
            backup_labels_dir = os.path.join(backup_dir, split, 'labels')
            backup_images_dir = os.path.join(backup_dir, split, 'images')
            os.makedirs(backup_labels_dir, exist_ok=True)
            os.makedirs(backup_images_dir, exist_ok=True)
            
            for label_filename in images_to_remove:
                shutil.move(os.path.join(labels_dir, label_filename), os.path.join(backup_labels_dir, label_filename))
                base_name = os.path.splitext(label_filename)[0]
                for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    src_image = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(src_image):
                        shutil.move(src_image, os.path.join(backup_images_dir, base_name + ext))
                        break
                total_removed += 1
                
    print(f"[SYSTEM] Dataset balancing complete. Relocated {total_removed} excess images.")

if __name__ == '__main__':
    undersample()
