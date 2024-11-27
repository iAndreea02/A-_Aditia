import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle


class Persoana:
    def __init__(self, id, nume, prenume, gen, data_programarii, salonul, tipul_problemei):
        self.id = id
        self.nume = nume
        self.prenume = prenume
        self.gen = gen
        self.data_programarii = data_programarii
        self.salonul = salonul
        self.tipul_problemei = tipul_problemei

    def __repr__(self):
        return (f"Persoana(ID: {self.id}, Nume: {self.nume}, Prenume: {self.prenume}, "
                f"Gen: {self.gen}, Data Programarii: {self.data_programarii}, "
                f"Salonul: {self.salonul}, Tipul Problemei: {self.tipul_problemei})")

# Crearea obiectelor Persoana
pers1 = Persoana(1, "Radu", "Andreea", "F", "2024-12-01", "Salon 1", "Control general")
pers2 = Persoana(2, "Grigoras", "Andrei", "M", "2024-12-02", "Salon 2", "Durere de spate")
pers3 = Persoana(3, "Cojan", "Alin", "M", "2024-12-03", "Salon 3", "Fractură")
persoane = [pers1, pers2, pers3]

def gaseste_dupa_prenume(prenume_cautat, persoane):
    # Normalizează prenumele pentru comparație
    prenume_cautat = prenume_cautat.strip().lower()
    return [pers for pers in persoane if pers.prenume.lower() == prenume_cautat]


# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (400, 300)}))
picam2.start()

# Initialize our variables
cv_scaler = 4  # this has to be a whole number
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

detected_names_count = {}  # Dictionary to count appearances of each name

def process_frame(frame):
    global face_locations, face_encodings, face_names

    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))

    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    return face_names  # Return face names instead of drawing on the frame

while True:
    # Capture a frame from camera
    frame = picam2.capture_array()

    # Process the frame and get the names of detected faces
    names_in_frame = process_frame(frame)

    # Update the count of detected names
    for name in names_in_frame:
        if name in detected_names_count:
            detected_names_count[name] += 1
        else:
            detected_names_count[name] = 1

    # Display FPS counter in console (optional)
    print(f"Detected: {', '.join(detected_names_count.keys()) if detected_names_count else 'No faces detected'}")

    # Show the video frame without any drawing
    cv2.imshow("Video", frame)

    # Stop after 6 seconds
    if time.time() - start_time > 6:
        break

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
picam2.stop()

# Print final detected names
print("\nFinal detected names:")
if detected_names_count:
    most_frequent_name = max(detected_names_count, key=detected_names_count.get)
    print(f"The person who appeared the most is: {most_frequent_name}")

    # Caută persoanele folosind prenumele detectat
    rezultate = gaseste_dupa_prenume(most_frequent_name, persoane)

    if rezultate:
        print("Rezultate găsite:", rezultate)
    else:
        print(f"Nu s-au găsit persoane cu prenumele '{most_frequent_name}'. Verifică datele!")
else:
    print("No faces detected.")
