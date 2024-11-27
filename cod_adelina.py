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

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 400)}))
picam2.start()

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()
fps = 0

# Control the scaling of the image to improve performance
cv_scaler = 3  # Rescaling factor to process fewer pixels

# Set the window size to 800x480
window_width = 800
window_height = 480

def calculate_angle(m1, m2):
    """Calculăm unghiul dintre două linii folosind pantele lor"""
    if 1 + m1 * m2 == 0:
        return 90.0  # Liniile sunt perpendiculare
    return np.arctan(abs((m1 - m2) / (1 + m1 * m2))) * (180.0 / np.pi)

def process_frame(frame):
    global face_locations, face_encodings, face_names

    # Resize the frame using cv_scaler to increase performance
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB for face recognition
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detect faces using face_recognition
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    return resized_frame

def draw_results(frame, face_landmarks):
    # Draw recognized faces and label them
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    # Use Mediapipe Face Mesh for additional face landmark detection (e.g., mouth, symmetry)
    if face_landmarks:
        for face_landmark in face_landmarks:
            # Extract landmarks
            left_corner = face_landmark.landmark[61]
            right_corner = face_landmark.landmark[291]
            forehead = face_landmark.landmark[10]
            chin = face_landmark.landmark[152]

            # Convert landmarks to pixel coordinates
            ih, iw, _ = frame.shape
            left_x, left_y = int(left_corner.x * iw), int(left_corner.y * ih)
            right_x, right_y = int(right_corner.x * iw), int(right_corner.y * ih)
            forehead_x, forehead_y = int(forehead.x * iw), int(forehead.y * ih)
            chin_x, chin_y = int(chin.x * iw), int(chin.y * ih)

            # Draw the points and symmetry line
            cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_x, right_y), 5, (0, 255, 0), -1)
            cv2.line(frame, (forehead_x, forehead_y), (chin_x, chin_y), (0, 255, 255), 2)

            # Calculate the slopes for symmetry detection
            delta_x_face = chin_x - forehead_x
            delta_y_face = chin_y - forehead_y
            slope_face = delta_y_face / delta_x_face if delta_x_face != 0 else float('inf')

            slope_left = (left_y - forehead_y) / (left_x - forehead_x) if (left_x - forehead_x) != 0 else float('inf')
            slope_right = (right_y - forehead_y) / (right_x - forehead_x) if (right_x - forehead_x) != 0 else float('inf')

            angle_left = calculate_angle(slope_face, slope_left)
            angle_right = calculate_angle(slope_face, slope_right)

            if angle_left < 9:
                print("Left mouth corner is lower")
            if angle_right < 9:
                print("Right mouth corner is lower")

    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Process the frame for face recognition
    processed_frame = process_frame(frame)

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Perform face mesh detection
    results = face_mesh.process(rgb_frame)
    face_landmarks = results.multi_face_landmarks if results.multi_face_landmarks else None

    # Draw face recognition and face mesh results
    display_frame = draw_results(processed_frame, face_landmarks)

    # Resize the frame to 800x480 before displaying it
    display_frame_resized = cv2.resize(display_frame, (window_width, window_height))

    # Calculate FPS
    current_fps = calculate_fps()

    # Display FPS
    cv2.putText(display_frame_resized, f"FPS: {current_fps:.1f}", (display_frame_resized.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with all results
    cv2.imshow('Video', display_frame_resized)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
picam2.stop()
