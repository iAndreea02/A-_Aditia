# epilepsy_detection.py

import cv2
import numpy as np
import time

# Variabile pentru logică
motion_history = []
max_motion_samples = 8
motion_threshold = 12  # Prag pentru mișcare intensă
spasm_threshold = 2  # Prag de timp pentru detectarea spasmelor (secunde)
spasm_start_time = None

def detect_spasmodic_movements(motion_history):
    """Detectează mișcări spasmodice pe baza istoricului de mișcare."""
    if len(motion_history) < 2:
        return False

    # Calculăm diferențele dintre pozițiile succesive
    diffs = np.diff(motion_history, axis=0)
    motion_intensity = np.linalg.norm(diffs, axis=1)

    # Detectăm mișcări intense
    intense_movements = motion_intensity > motion_threshold
    return np.sum(intense_movements) > len(intense_movements) * 0.5

def track_motion(landmarks):
    """Monitorizează mișcarea capului folosind punctele cheie."""
    global motion_history

    # Selectăm punctele importante: nas, frunte, bărbie
    key_points = [landmarks[1], landmarks[10], landmarks[152]]  # Nas, frunte, bărbie

    # Calculăm centrul punctelor
    center = np.mean(key_points, axis=0)

    # Adăugăm la istoricul mișcării
    if len(motion_history) >= max_motion_samples:
        motion_history.pop(0)
    motion_history.append(center)

    return center

def detect_epilepsy(frame):
    """Detectează spasme care ar putea semnala o criză epileptică."""
    global spasm_start_time
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

    return frame
