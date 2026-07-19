import cv2
import requests

API_URL = "http://localhost:8000/verify"

def live_verification_client():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("[MAC FIX] If you are on a Mac, go to System Settings -> Privacy & Security -> Camera")
        print("[MAC FIX] Make sure your Terminal or VS Code is checked to allow camera access!")
        return
        
    print("\n[SYSTEM] Connecting to AegisAuth API for live verification...")
    print("Ensure the API is running locally via 'python main.py'.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Encode frame to JPEG format in memory
        _, buffer = cv2.imencode('.jpg', frame)
        
        try:
            # Send standard HTTP POST to the API
            response = requests.post(
                API_URL, 
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=1.5 # short timeout so UI doesn't hang forever
            )
            
            if response.status_code == 200:
                data = response.json()
                faces = data.get("faces", [])
                
                for face in faces:
                    box = face["bounding_box"]
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    status = face["status"]
                    liveness = face["liveness_score"]
                    identity = face["identity"]
                    
                    if status == "Verified":
                        color = (0, 255, 0) # Green
                        label = f"Verified: {identity} (Liveness: {liveness:.2f})"
                    elif status == "Unregistered":
                        color = (0, 165, 255) # Orange
                        label = f"Unregistered (Liveness: {liveness:.2f})"
                    else:
                        color = (0, 0, 255) # Red
                        label = f"Spoof Detected (Liveness: {liveness:.2f})"
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"API Error: {response.status_code}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except requests.exceptions.RequestException:
             cv2.putText(frame, "API Disconnected - Please start server", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('AegisAuth Client Dashboard', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_verification_client()