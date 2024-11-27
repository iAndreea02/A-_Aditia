import face_recognition
import cv2
import numpy as np
import time
import pickle
import mediapipe as mp

# Încarcă encodările faciale
print("[INFO] loading encodings...")
with open("../encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Inițializează camera
cap = cv2.VideoCapture(0)

# Initializează MediaPipe pentru detectarea feței
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Variabile pentru logică
cv_scaler = 3  # Scăderea dimensiunii imaginii pentru performanță
spasm_start_time = None
motion_history = []
max_motion_samples = 8
motion_threshold = 12  # Prag pentru mișcare intensă
spasm_threshold = 2  # Prag de timp pentru detectarea spasmelor (secunde)
person_identified = False  # Flag pentru a verifica dacă persoana a fost identificată

def process_frame(frame):
    global face_locations, face_encodings, face_names
    
    # Redimensionăm imaginea pentru performanță
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Conversie din BGR în RGB
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Detectarea feței și extragerea encodării
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
    
    return resized_frame

def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
    
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

def detect_spasmodic_movements(motion_history):
    """Detectează mișcări spasmodice bazate pe istoricul vectorilor."""
    if len(motion_history) < 2:
        return False
    diffs = np.diff(motion_history, axis=0)
    motion_intensity = np.linalg.norm(diffs, axis=1)
    intense_movements = motion_intensity > motion_threshold
    return np.sum(intense_movements) > len(intense_movements) * 0.5

def track_motion(landmarks):
    """Monitorizează mișcarea capului folosind punctele cheie."""
    global motion_history
    key_points = [landmarks[1], landmarks[10], landmarks[152]]  # Nas, mijloc frunte, bărbie
    center = np.mean(key_points, axis=0)
    if len(motion_history) >= max_motion_samples:
        motion_history.pop(0)
    motion_history.append(center)
    return center

print("Detectare epilepsie pornita... Apasa 'q' pentru a opri.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Eroare la capturarea fluxului video.")
        break

    # Procesăm imaginea pentru recunoașterea feței
    processed_frame = process_frame(frame)

    # Dacă persoana a fost identificată, oprim filmarea
    if not person_identified and face_names:
        name = face_names[0]
        if name != "Unknown":
            print(f"[INFO] Persoana identificată: {name}")
            person_identified = True
            cap.release()  # Închide camera după identificare
            break  # Opriți procesarea video

    # Desenăm rezultatele
    display_frame = draw_results(processed_frame)

    # Calculăm FPS
    current_fps = calculate_fps()

    # Afișăm FPS-ul
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afișăm fluxul video
    cv2.imshow("Detectare Epilepsie", display_frame)

    # Dacă persoana a fost identificată, urmărim mișcările pentru detectarea crizei
    if person_identified:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        height, width, _ = frame.shape
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(landmark.x * width), int(landmark.y * height))
                             for landmark in face_landmarks.landmark]

                track_motion(landmarks)
                
                if detect_spasmodic_movements(motion_history):
                    if spasm_start_time is None:
                        spasm_start_time = time.time()

                    elapsed_time = time.time() - spasm_start_time
                    cv2.putText(frame, f"CRIZA DETECTATa: {int(elapsed_time)} sec", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if elapsed_time > spasm_threshold:
                        cv2.putText(frame, "ALERTA: Posibila criza epileptica!", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    spasm_start_time = None

    # Închide aplicația când apăsați 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
