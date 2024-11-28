import cv2
import mediapipe as mp
import numpy as np
import time
from picamera2 import Picamera2

# Inițializare Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inițializare Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())  # Configurare pentru captură video
picam2.start()

# Controlul cadrelor procesate pe secundă
fps = 30
frame_time = 1.0 / fps
last_frame_time = 0

# Dimensiuni fereastră video
frame_width, frame_height = 400, 300

def calculate_angle(m1, m2):
    """Calculăm unghiul dintre două linii folosind pantele lor"""
    if 1 + m1 * m2 == 0:
        return 90.0  # Liniile sunt perpendiculare
    return np.arctan(abs((m1 - m2) / (1 + m1 * m2))) * (180.0 / np.pi)

while True:
    # Capturăm un cadru de la camera Picamera2
    frame = picam2.capture_array()

    # Verificăm formatul imaginii
   # print(f"Image format: {frame.shape}")  # Afișează dimensiunile imaginii capturate

    # Redimensionăm cadrul la dimensiunea dorită
    frame_resized = cv2.resize(frame, (frame_width, frame_height))
     #Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Verificăm dacă timpul curent este mai mare decât timpul pentru un cadru
    if time.time() - last_frame_time >= frame_time:
        last_frame_time = time.time()

        # Detectarea feței și a caracteristicilor
        results = face_mesh.process(rgb_resized_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Landmark-uri pentru colțurile gurii
                left_corner = face_landmarks.landmark[61]
                right_corner = face_landmarks.landmark[291]

                # Landmark-uri pentru frunte și bărbie pentru axa de simetrie
                forehead = face_landmarks.landmark[10]
                chin = face_landmarks.landmark[152]

                # Conversie la coordonate pixel
                ih, iw, _ = rgb_resized_frame.shape
                left_x, left_y = int(left_corner.x * iw), int(left_corner.y * ih)
                right_x, right_y = int(right_corner.x * iw), int(right_corner.y * ih)

                forehead_x, forehead_y = int(forehead.x * iw), int(forehead.y * ih)
                chin_x, chin_y = int(chin.x * iw), int(chin.y * ih)

                # Calculăm panta axei feței
                delta_x_face = chin_x - forehead_x
                delta_y_face = chin_y - forehead_y
                if delta_x_face != 0:
                    slope_face = delta_y_face / delta_x_face
                else:
                    slope_face = float('inf')

                # Calculăm pantele pentru fiecare colț al gurii față de axa de simetrie
                slope_left = (left_y - forehead_y) / (left_x - forehead_x) if (left_x - forehead_x) != 0 else float('inf')
                slope_right = (right_y - forehead_y) / (right_x - forehead_x) if (right_x - forehead_x) != 0 else float('inf')

                # Calculăm unghiurile dintre fiecare colț al gurii și axa feței
                angle_left = calculate_angle(slope_face, slope_left)
                angle_right = calculate_angle(slope_face, slope_right)

                # Determinăm dacă vreun colț este lăsat
                if angle_left < 9:
                    print("Left mouth corner is lower")
                if angle_right < 9:
                    print("Right mouth corner is lower")

        # Afișare video cu dimensiunile setate
        cv2.imshow("Video Feed - Mouth Corner Detection", rgb_resized_frame)

    # Verificare dacă utilizatorul a apăsat 'q' sau a închis fereastra
    if cv2.getWindowProperty("Video Feed - Mouth Corner Detection", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberare resurse
picam2.stop()
cv2.destroyAllWindows()
