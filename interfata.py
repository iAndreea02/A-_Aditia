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
from picamera2 import Picamera2


class FacialRecognitionApp:
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
            command=self.start_recognition_thread
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

    def start_recognition_thread(self):
        """Pornirea procesului de recunoaștere facială într-un fir separat."""
        self.running = True
        self.message_label.config(text="Recunoaștere facială în curs...")
        threading.Thread(target=self.start_recognition, daemon=True).start()

    def start_recognition(self):
        """Funcția de procesare a video-ului și recunoaștere facială."""
        print("[INFO] Loading encodings...")
        with open("encodings.pickle", "rb") as f:
            data = pickle.loads(f.read())
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]

        # Configurare cameră
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (400, 300)}))
        self.picam2.start()

        detected_names_count = {}
        start_time = time.time()
        cv_scaler = 4

        while self.running and time.time() - start_time < 5:  # Limită de 10 secunde
            frame = self.picam2.capture_array()
            resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
            rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Detectare fețe
            face_locations = face_recognition.face_locations(rgb_resized_frame)
            face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

            # Identificare fețe
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

            # Contorizare
            for name in face_names:
                if name in detected_names_count:
                    detected_names_count[name] += 1
                else:
                    detected_names_count[name] = 1

            # Afișare în `Label`
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.root.update()

        self.picam2.stop()

        # Rezultatele detectării
        print("\nFinal detected names:")
        if detected_names_count:
            most_frequent_name = max(detected_names_count, key=detected_names_count.get)
            print(f"Bun venit, {most_frequent_name}!")

            # Afișează informațiile despre programare
            if most_frequent_name == 'Unknown':
                self.result_label.config(
                    text=f"Persoana necunoscută!",
                    fg="red"
                )
            else:
                # Căutăm datele persoanei în dicționar
                patient_info = self.patient_data.get(most_frequent_name)
                if patient_info:
                    self.result_label.config(
                        text=f"{most_frequent_name}\nProgramare: {patient_info['data_programarii']}\nSecția: {patient_info['Etaj/Sectie']}\nTipul problemei: {patient_info['tipul_problemei']}",
                        fg="green"
                    )
                else:
                    self.result_label.config(
                        text=f"Informațiile nu sunt disponibile.",
                        fg="orange"
                    )
        else:
            print("No faces detected.")
            self.result_label.config(
                text="Nu s-au detectat fețe în cadru.",
                fg="red"
            )

        # Oprește fluxul video
        self.running = False

    def on_close(self):
        """Închidere corectă a aplicației."""
        self.running = False
        if self.picam2:
            self.picam2.stop()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()