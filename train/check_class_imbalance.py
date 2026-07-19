import os
import matplotlib.pyplot as plt

def check_imbalance():
    base_dir = "/content/face-anti-spoofing-detection-2" if os.path.exists("/content") else "./face-anti-spoofing-detection-2"
    splits = ['train', 'valid', 'test']
    
    class_names = {0: 'Fake', 1: 'Real'}
    class_counts = {0: 0, 1: 0}
    
    for split in splits:
        labels_dir = os.path.join(base_dir, split, 'labels')
        if not os.path.exists(labels_dir): continue
            
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(labels_dir, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id in class_counts:
                                class_counts[class_id] += 1
                                
    print("\n--- Current Class Distribution ---")
    for class_id, count in class_counts.items():
        print(f"{class_names[class_id]} (Class {class_id}): {count}")
    print("----------------------------------\n")
        
    names = [class_names[cls_id] for cls_id in class_counts.keys()]
    counts = [class_counts[cls_id] for cls_id in class_counts.keys()]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, counts, color=['red', 'blue'])
    plt.title('Class Distribution (Face Anti-Spoofing)')
    plt.xlabel('Classes')
    plt.ylabel('Number of Bounding Boxes')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), ha='center', va='bottom')
        
    plot_path = os.path.join(base_dir, 'class_imbalance_plot.png')
    plt.savefig(plot_path)
    print(f"[SYSTEM] Plot saved to {plot_path}")

if __name__ == "__main__":
    check_imbalance()
