import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import time
import pickle
from picamera2 import Picamera2
 
# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]
 
# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 400)}))
picam2.start()
 
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)
 
# Camera variables
cap = picam2
 
# Variables for spasms detection
spasm_start_time = None
motion_history = []
max_motion_samples = 8
motion_threshold = 12  # Threshold for intense motion
spasm_threshold = 2  # Threshold for spasm duration in seconds
 
# Variables for FPS calculation
frame_count = 0
start_time = time.time()
fps = 0
 
# Function to detect spasmodic movements
def detect_spasmodic_movements(motion_history):
    """Detects spasmodic movements based on motion history."""
    if len(motion_history) < 2:
        return False
 
    diffs = np.diff(motion_history, axis=0)
    motion_intensity = np.linalg.norm(diffs, axis=1)
 
    intense_movements = motion_intensity > motion_threshold
    return np.sum(intense_movements) > len(intense_movements) * 0.5
 
# Function to track head motion
def track_motion(landmarks):
    """Tracks head motion based on key face landmarks."""
    global motion_history
 
    # Key points: nose, forehead, chin
    key_points = [landmarks[1], landmarks[10], landmarks[152]]  # Nose, forehead, chin
 
    # Calculate the center of key points
    center = np.mean(key_points, axis=0)
 
    # Update motion history
    if len(motion_history) >= max_motion_samples:
        motion_history.pop(0)
    motion_history.append(center)
 
    return center
 
# Function to calculate FPS
def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps
 
# Main loop
while True:
    # Capture frame from the camera
    frame = picam2.capture_array()
 
    # Reduce the frame size for faster processing
    resized_frame = cv2.resize(frame, (320, 200))  # Reduce resolution to 320x200 for faster processing
 
    # Convert frame for face recognition
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
 
    # Every n-th frame (every 3rd or 5th frame) process face recognition
    if frame_count % 5 == 0:
        # Detect faces using face_recognition
        face_locations = face_recognition.face_locations(rgb_resized_frame)
        face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)
 
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
 
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
 
    # Perform face mesh detection (process every frame)
    results = face_mesh.process(rgb_frame)
    face_landmarks = results.multi_face_landmarks if results.multi_face_landmarks else None
 
    if face_landmarks:
        for face_landmark in face_landmarks:
            # Extract landmarks
            landmarks = [(int(landmark.x * resized_frame.shape[1]), int(landmark.y * resized_frame.shape[0]))
                         for landmark in face_landmark.landmark]
 
            # Track head motion
            track_motion(landmarks)
 
            # Detect spasms
            if detect_spasmodic_movements(motion_history):
                if spasm_start_time is None:
                    spasm_start_time = time.time()
 
                elapsed_time = time.time() - spasm_start_time
                cv2.putText(resized_frame, f"CRIZA DETECTATa: {int(elapsed_time)} sec", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
                if elapsed_time > spasm_threshold:
                    cv2.putText(resized_frame, "ALERTA: Posibila criza epileptica!", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                spasm_start_time = None
 
    # Draw face rectangles and names for recognized faces
    if face_locations:
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 1  # Adjust coordinates to current frame size
            right *= 1
            bottom *= 1
            left *= 1
            cv2.rectangle(resized_frame, (left, top), (right, bottom), (244, 42, 3), 3)  # Orange box for face
            cv2.rectangle(resized_frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(resized_frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
 
    # Calculate FPS
    current_fps = calculate_fps()
 
    # Display FPS
    cv2.putText(resized_frame, f"FPS: {current_fps:.1f}", (resized_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    # Resize the frame for display
    display_frame_resized = cv2.resize(resized_frame, (800, 480))
 
    # Show the frame
    cv2.imshow("Detectare epilepsie", display_frame_resized)
 
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break
 
# Cleanup
cv2.destroyAllWindows()
picam2.stop()
