import os
import cv2
import time
import pickle
import face_recognition
from datetime import datetime
from picamera2 import Picamera2
 
# Funcție pentru a crea un folder pentru fiecare utilizator
def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder
 
# Funcție pentru a captura automat 50 de poze
def capture_photos(name):
    folder = create_folder(name)
    # Inițializarea camerei
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()
 
    # Permite camerei să se încălzească
    time.sleep(2)
 
    photo_count = 0
    print(f"Capturarea automată a pozelor pentru {name}. Capturăm 50 de poze...")
    while photo_count < 50:
        # Capturează frame-ul de la camera Pi
        frame = picam2.capture_array()
        # Salvează fiecare poză cu un timestamp unic
        photo_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, frame)
        print(f"Foto {photo_count} salvată: {filepath}")
        # Așteaptă 1 secundă înainte de a captura următoarea poză
        time.sleep(1)
    # Curățare
    cv2.destroyAllWindows()
    picam2.stop()
    print(f"Capturarea automată a fost completată. {photo_count} poze salvate pentru {name}.")
 
# Funcție pentru a salva encodările faciale
def save_face_encodings():
    print("[INFO] Încep procesarea fețelor...")
    imagePaths = list(paths.list_images("dataset"))
    knownEncodings = []
    knownNames = []
 
    for (i, imagePath) in enumerate(imagePaths):
        print(f"[INFO] Procesare imagine {i + 1}/{len(imagePaths)}")
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
 
    print("[INFO] Salvarea encodărilor...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open("encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))
 
    print("[INFO] Procesul de antrenament s-a încheiat. Encodările au fost salvate în 'encodings.pickle'.")
 
# Funcția principală
def main():
    name = input("Introduceti numele utilizatorului pentru profil: ")
 
    # Capturează poze pentru profilul respectiv
    capture_photos(name)
 
    # După ce pozele au fost capturate, salvează encodările faciale
    save_face_encodings()
 
    print("Recunoașterea facială a avut succes! Profilul a fost creat.")
 
if __name__ == "__main__":
    main()