import tkinter as tk
import customtkinter as ctk
import cv2
import time
import numpy as np
import pickle
import mediapipe as mp
from picamera2 import Picamera2
import facial_recognition  # Importa funcția facial_recognition.py

# Fereastra pop-up pentru avertisment
def show_warning_popup():
    def on_ok_button_click():
        warning_window.destroy()

    warning_window = tk.Toplevel()
    warning_window.title("Avertisment")
    warning_window.geometry("400x200")
    warning_window.configure(bg="#ffffff")

    warning_label = tk.Label(
        warning_window,
        text="Există semne de risc AVC! Verificați imediat!",
        font=("Comic Sans MS", 14, "bold"),
        bg="#ffffff",
        fg="#FF0000"
    )
    warning_label.pack(pady=40)

    ok_button = ctk.CTkButton(
        warning_window,
        text="OK",
        height=40,
        width=200,
        font=("Helvetica", 14, "bold"),
        text_color="white",
        fg_color="#007BFF",
        hover_color="#0056b3",
        corner_radius=20,
        command=on_ok_button_click
    )
    ok_button.pack(pady=20)

    warning_window.mainloop()

# Funcția pentru recunoașterea facială
def facial_recognition():
    # Restul codului tău pentru recunoașterea facială și calculul unghiului
    # ....

    def calculate_angle(m1, m2):
        """Calculăm unghiul dintre două linii folosind pantele lor"""
        if 1 + m1 * m2 == 0:
            return 90.0  # Liniile sunt perpendiculare
        return np.arctan(abs((m1 - m2) / (1 + m1 * m2))) * (180.0 / np.pi)

    def draw_results(frame, face_landmarks):
        # Codul pentru desenarea landmarkurilor faciale
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

                print(f"left_corner: {left_x}, {left_y}, right_corner: {right_x}, {right_y}")  # Adaugă print pentru a vedea datele

                # Calculăm pantele pentru simetrie
                delta_x_face = chin_x - forehead_x
                delta_y_face = chin_y - forehead_y
                slope_face = delta_y_face / delta_x_face if delta_x_face != 0 else float('inf')

                slope_left = (left_y - forehead_y) / (left_x - forehead_x) if (left_x - forehead_x) != 0 else float('inf')
                slope_right = (right_y - forehead_y) / (right_x - forehead_x) if (right_x - forehead_x) != 0 else float('inf')

                angle_left = calculate_angle(slope_face, slope_left)
                angle_right = calculate_angle(slope_face, slope_right)

                # Dacă unghiul este prea mic, se consideră că este un risc
                if angle_left < 9 or angle_right < 9:
                    print("Risc detectat: Afișez fereastra pop-up")  # Verifică dacă acest mesaj apare
                    show_warning_popup()  # Afișează fereastra de avertisment

        return frame

    # Restul codului pentru procesarea imaginii și camerei
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (400, 300)}))
    picam2.start()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        face_landmarks = results.multi_face_landmarks if results.multi_face_landmarks else None
        display_frame = draw_results(frame, face_landmarks)

        # Afișăm frame-ul procesat
        display_frame_resized = cv2.resize(display_frame, (400, 300))
        cv2.imshow('Video', display_frame_resized)

        # Ieșim din buclă dacă se apasă tasta 'q'
        if cv2.waitKey(1) == ord("q"):
            break

    # Curățarea resurselor
    cv2.destroyAllWindows()
    picam2.stop()

# Fereastra de bun venit
def show_initial_window():
    def on_start_button_click():
        message_label.pack_forget()
        recunoastere_button.pack_forget()
        start_label.pack(pady=20)
        facial_recognition()  # Apelăm funcția facial_recognition

    initial_window = tk.Tk()
    initial_window.title("Bun venit!")
    initial_window.geometry("480x800")
    initial_window.configure(bg="#ffffff")

    message_label = tk.Label(
        initial_window,
        text="Bine ai venit la aplicația noastră!\nApasă OK pentru a începe.",
        font=("Comic Sans MS", 14, "bold"),
        bg="#ffffff",
        fg="#007BFF"
    )
    message_label.pack(pady=40)

    recunoastere_button = ctk.CTkButton(
        initial_window,
        text="Începe",
        height=40,
        width=200,
        font=("Helvetica", 14, "bold"),
        text_color="white",
        fg_color="#007BFF",
        hover_color="#0056b3",
        corner_radius=20,
        command=on_start_button_click
    )
    recunoastere_button.pack(pady=20)

    start_label = tk.Label(
        initial_window,
        text="Acțiunea a fost începută!",
        font=("Comic Sans MS", 14, "bold"),
        bg="#ffffff",
        fg="#007BFF"
    )

    initial_window.mainloop()

# Începerea aplicației
show_initial_window()
