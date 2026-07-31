import os
import re
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "production.log")

def analyze_model_drift(log_file=DEFAULT_LOG_FILE):
    """
    Parses the production logs to analyze liveness score trends over time.
    Detects potential 'Data Drift' if average confidence scores for 'Verified' requests drop significantly.
    """
    try:
        with open(log_file, "r") as f:
            logs = f.readlines()
    except FileNotFoundError:
        print("[ERROR] Production log file not found. No inferences to analyze yet.")
        return

    # Regex to extract telemetry
    pattern = re.compile(r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*Status: (?P<status>\w+).*Liveness: (?P<liveness>[\d\.]+)")
    
    liveness_scores = []
    
    for line in logs:
        match = pattern.search(line)
        if match:
            # We specifically care about liveness drift for valid/real inputs over time.
            if match.group("status") == "Verified":
                liveness_scores.append(float(match.group("liveness")))
                
    if len(liveness_scores) < 10:
        print("[INFO] Not enough telemetry data to statistically study drift (need at least 10 Verified inferences).")
        print(f"Currently have {len(liveness_scores)} verified inferences logged.")
        return
        
    # Simple Drift Study: Compare first half of data to second half of data
    midpoint = len(liveness_scores) // 2
    baseline_scores = liveness_scores[:midpoint]
    recent_scores = liveness_scores[midpoint:]
    
    baseline_mean = np.mean(baseline_scores)
    recent_mean = np.mean(recent_scores)
    
    drift_delta = recent_mean - baseline_mean
    
    print("\n--- 📊 Model Drift Analysis Report ---")
    print(f"Total Verified Inferences Analyzed: {len(liveness_scores)}")
    print(f"Baseline Liveness Mean (Older): {baseline_mean:.4f}")
    print(f"Recent Liveness Mean (Newer): {recent_mean:.4f}")
    print(f"Delta: {drift_delta:+.4f}")
    
    if drift_delta < -0.05:
        print("\n[WARNING] Significant Model Drift Detected!")
        print("Recent liveness scores have dropped by more than 5%.")
        print("This indicates environmental data drift (e.g., new webcams or lighting conditions) or novel spoofing attacks.")
        print("Recommendation: Extract recent misclassifications and trigger retraining pipeline.")
    else:
        print("\n[OK] No significant drift detected. Model is performing stably in production.")

if __name__ == "__main__":
    analyze_model_drift()
