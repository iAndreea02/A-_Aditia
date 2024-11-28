import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import time
import threading
import pickle
import customtkinter as ctk
import mediapipe as mp
from picamera2 import Picamera2


class AVCApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recunoaștere Facială")
        self.root.geometry("480x800")
        self.root.configure(bg="#ffffff")
        
        # Panoul cu mesajul de întâmpinare
        self.message_panel = tk.Frame(root, bg="#ffffff")
        self.message_panel.pack(pady=10, fill="both", expand=False)

        # Mesajul de întâmpinare
        self.message_label = tk.Label(
            root,
            text="Bine ai venit la Clinica!\nApasă OK pentru a începe scanarea \n pentru a afla informatii si \nDIAGNOSTICARE \n pentru a primii un diagnostic urgent!",
            font=("Comic Sans MS", 24, "bold"),
            bg="#ffffff",
            fg="#007BFF"
        )
        self.message_label.pack(pady=5, fill="both", expand=False)

        # Butonul pentru a începe recunoașterea facială
        self.recunoastere_button = ctk.CTkButton(
            root,
            text="OK",
            height=40,
            width=200,
            font=("Helvetica", 14, "bold"),
            text_color="white",
            fg_color="#007BFF",
            hover_color="#0056b3",
            corner_radius=20,
            command=self.start_diagnostic_thread
        )
        self.recunoastere_button.pack(pady=20)

        # Etichetă pentru afișarea fluxului video
        self.video_label = tk.Label(root, bg="#ffffff")
        self.video_label.pack(pady=20)

        # Etichetă pentru afișarea rezultatului
        self.result_label = tk.Label(
            root,
            text="",
            font=("Helvetica", 16, "bold"),
            bg="#ffffff",
            fg="black"
        )
        self.result_label.pack(pady=10)

        # Dicționarul cu datele pacienților
        self.patient_data = {
            "Andreea": {'id': 1, 'nume': 'Radu', 'prenume': 'Andreea', 'gen': 'F', 'data_programarii': '2024-12-01', 'Etaj/Sectie': 'Etajul 5, Sectia Medicina generala', 'tipul_problemei': 'Control general'},
            "Andrei": {'id': 2, 'nume': 'Grigoras', 'prenume': 'Andrei', 'gen': 'M', 'data_programarii': '2024-12-02', 'Etaj/Sectie': 'Etajul 7, Sectia Ortopedie', 'tipul_problemei': 'Durere de spate'},
            "Alin": {'id': 3, 'nume': 'Cojan', 'prenume': 'Alin', 'gen': 'M', 'data_programarii': '2024-12-03', 'Etaj/Sectie': 'Etajul 6, Sectia Ortopedie', 'tipul_problemei': 'Fractură'}
        }

        # Variabile de control
        self.running = False
        self.picam2 = None
        self.time_left = 0
        self.time_right = 0
        self.alert_sent = False


    def calculate_angle(self, m1, m2):
        if 1 + m1 * m2 == 0:
            return 90.0  # Liniile sunt perpendiculare
        return np.arctan(abs((m1 - m2) / (1 + m1 * m2))) * (180.0 / np.pi)
    
    def start_diagnostic_thread(self):
        self.running = True
        self.message_label.config(text="Diagnosticare în curs...")
        threading.Thread(target=self.run_diagnostic, daemon=True).start()

    def run_diagnostic(self):
        # Inițializează FaceMesh din Mediapipe
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Inițializează camera
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_still_configuration())
        self.picam2.start()

        fps = 30
        frame_time = 1.0 / fps
        last_frame_time = 0
        frame_width, frame_height = 400, 300
        start_time = time.time()
        max_time = 20  # Timp maxim pentru diagnosticare (20 secunde)
        alert_sent = False  # Variabilă pentru a trimite alertă doar o dată

        while self.running and time.time() - start_time < max_time:
            # Capturează o imagine din cameră
            frame = self.picam2.capture_array()

            frame_resized = cv2.resize(frame, (frame_width, frame_height))
            rgb_resized_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            if time.time() - last_frame_time >= frame_time:
                last_frame_time = time.time()

                # Detectare fețe și colțuri ale gurii
                results = face_mesh.process(rgb_resized_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        left_corner = face_landmarks.landmark[61]
                        right_corner = face_landmarks.landmark[291]
                        forehead = face_landmarks.landmark[10]
                        chin = face_landmarks.landmark[152]

                        ih, iw, _ = rgb_resized_frame.shape
                        left_x, left_y = int(left_corner.x * iw), int(left_corner.y * ih)
                        right_x, right_y = int(right_corner.x * iw), int(right_corner.y * ih)
                        forehead_x, forehead_y = int(forehead.x * iw), int(forehead.y * ih)
                        chin_x, chin_y = int(chin.x * iw), int(chin.y * ih)

                        delta_x_face = chin_x - forehead_x
                        delta_y_face = chin_y - forehead_y
                        if delta_x_face != 0:
                            slope_face = delta_y_face / delta_x_face
                        else:
                            slope_face = float('inf')

                        slope_left = (left_y - forehead_y) / (left_x - forehead_x) if (left_x - forehead_x) != 0 else float('inf')
                        slope_right = (right_y - forehead_y) / (right_x - forehead_x) if (right_x - forehead_x) != 0 else float('inf')

                        angle_left = self.calculate_angle(slope_face, slope_left)
                        angle_right = self.calculate_angle(slope_face, slope_right)

                        # Verifică dacă colțul gurii este mai jos decât o valoare limită
                        if angle_left < 9:
                            self.time_left += 1
                        else:
                            self.time_left = 0  # Resetăm timpul dacă colțul gurii revine la normal

                        if angle_right < 9:
                            self.time_right += 1
                        else:
                            self.time_right = 0  # Resetăm timpul dacă colțul gurii revine la normal

                        # Dacă colțul gurii este lăsat mai mult de 6 secunde, trimite o alertă
                        if self.time_left > 6 and not alert_sent:
                            alert_sent = True
                            self.result_label.config(text="Atenție!\n Posibil AVC! \n Consultați un medic imediat!")
                            self.message_label.config(text="Diagnosticare în curs... Persoana suferă de AVC.")
                            break

                        if self.time_right > 6 and not alert_sent:
                            alert_sent = True
                            self.result_label.config(text="Atenție!\n Posibil AVC! \n Consultați un medic imediat!")
                            self.message_label.config(text="Diagnosticare în curs... Persoana suferă de AVC.")
                            break

            # Afișare în Label
            frame_bgr = cv2.cvtColor(rgb_resized_frame, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(frame_bgr)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.root.update()

        # Oprirea camerei
        self.picam2.stop()
        if not alert_sent:
            #self.result_label.config(text="Diagnosticare finalizată!")
            self.message_label.config(text="Diagnosticare încheiată. Consultați medicul pentru detalii.")

    def on_close(self):
        """Închidere corectă a aplicației."""
        self.running = False
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = AVCApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
