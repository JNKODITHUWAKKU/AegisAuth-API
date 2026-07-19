import cv2
import requests
import time

API_URL = "http://localhost:8000/enroll"

def enroll_via_api():
    student_id = input("Enter Student ID to enroll: ").strip()
    if not student_id:
        print("[ERROR] Invalid Student ID. Aborting.")
        return

    print(f"\n--- Initializing AegisAuth API Enrollment for Student: {student_id} ---")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("[MAC FIX] If you are on a Mac, go to System Settings -> Privacy & Security -> Camera")
        print("[MAC FIX] Make sure your Terminal or VS Code is checked to allow camera access!")
        return

    frames_captured = []
    ENROLL_ANGLES = 30
    CAPTURE_DELAY = 0.15
    last_capture_time = time.time()
    
    print("Please look at the camera. Capturing images...")
    try:
        while len(frames_captured) < ENROLL_ANGLES:
            ret, frame = cap.read()
            if not ret: continue
            
            # Live UI Feedback Overlay
            cv2.putText(frame, f"Capturing: {len(frames_captured)}/{ENROLL_ANGLES} (Press Q to cancel)", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("AegisAuth API Enrollment Client", frame)
            
            current_time = time.time()
            if current_time - last_capture_time >= CAPTURE_DELAY:
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frames_captured.append(buffer.tobytes())
                last_capture_time = time.time()
                print(f"Captured frame {len(frames_captured)}/{ENROLL_ANGLES}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[ABORT] Enrollment terminated by user.")
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nSending securely encrypted frames to AegisAuth API for vector extraction...")
    
    # Construct multipart/form-data for files
    files = [("files", (f"frame_{i}.jpg", frame_bytes, "image/jpeg")) for i, frame_bytes in enumerate(frames_captured)]
    data = {"student_id": student_id}
    
    try:
        response = requests.post(API_URL, data=data, files=files)
        if response.status_code == 200:
            result = response.json()
            print(f"\n[SUCCESS] Profile synced via API:")
            print(f"Student: {result['student_id']}")
            print(f"Vectors stored: {result['anchors_saved']}")
        else:
            print(f"\n[ERROR] API rejected the request ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"\n[CRITICAL ERROR] Could not connect to AegisAuth API: {e}")
        print("Please ensure the API is running (python main.py).")

if __name__ == "__main__":
    enroll_via_api()