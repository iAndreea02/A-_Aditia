import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess  # Modul pentru a rula scripturi externe

# Crează fereastra principală
root = tk.Tk()
root.title("Meniu Start")
root.geometry("480x800")  # Dimensiunea ferestrei (stil portret)

# Încarcă imaginea folosind Pillow
img = Image.open("back1.png")
img = img.resize((480, 800), Image.Resampling.LANCZOS)  # Redimensionare pentru a se potrivi ferestrei
background_image = ImageTk.PhotoImage(img)

# Adaugă imaginea de fundal
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)  # Întinde imaginea pe întreaga fereastră

# Obține culoarea fundalului pentru iluzia de transparență
def get_background_color(x, y):
    pixel = img.getpixel((x, y))  # Citește culoarea pixelului la poziția dată
    return f'#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}'

# Coordonatele titlului
title_x = int(480 * 0.5)  # Coordonata X centrală
title_y = int(800 * 0.05)  # Coordonata Y la 5% din înălțime
bg_color = get_background_color(title_x, title_y)

# Adaugă un titlu stilizat
title_label = tk.Label(
    root,
    text="Medscan AI",
    font=("Helvetica", 40, "bold italic"),
    fg="#FFD700",  # Text auriu
    bg=bg_color  # Fundalul sincronizat cu imaginea
)
title_label.place(relx=0.5, rely=0.05, anchor="center")

# Crează funcțiile pentru butoane
def button1_action():
    # Redirecționează către scriptul1.py
    subprocess.run(["python3", "interfata.py"])  # Înlocuiește cu calea către scriptul tău
    root.quit()  # Închide fereastra principală

def button2_action():
    # Redirecționează către scriptul2.py
    subprocess.run(["python3", "interfata_AVC.py"])  # Înlocuiește cu calea către scriptul tău
    root.quit()  # Închide fereastra principală

# Stil personalizat pentru butoane
def create_custom_button(text, command, y_position):
    # Cadru exterior pentru rotunjire
    frame = tk.Frame(root, bg="#FFD700", bd=2)
    frame.place(relx=0.5, rely=y_position, anchor="center", width=250, height=50)
    
    # Fundal gradient pentru buton
    canvas = tk.Canvas(frame, width=250, height=50, bd=0, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    # Setează culoarea de fundal inițială la albastru închis
    gradient = canvas.create_rectangle(0, 0, 250, 50, fill="#003366", outline="#DCEFF8")
    canvas.tag_lower(gradient)

    # Textul butonului
    button = tk.Label(
        frame,
        text=text,
        font=("Helvetica", 14, "bold"),
        bg="#003366",
        fg="white",
        cursor="hand2"
    )
    button.place(relx=0.5, rely=0.5, anchor="center")

    # Animație la hover
    def on_enter(event):
        canvas.itemconfig(gradient, fill="#336699")  # Albastru deschis când treci cu mouse-ul
        button.config(bg="#336699")

    def on_leave(event):
        canvas.itemconfig(gradient, fill="#003366")  # Albastru închis când ieși cu mouse-ul
        button.config(bg="#003366")

    def on_press(event):
        canvas.itemconfig(gradient, fill="#99CCFF")  # Albastru deschis când este apăsat
        button.config(bg="#99CCFF")

    def on_release(event):
        canvas.itemconfig(gradient, fill="#003366")  # Revine la albastru închis după ce este eliberat
        button.config(bg="#003366")

    # Bind pentru hover și apăsare
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    button.bind("<ButtonPress-1>", on_press)
    button.bind("<ButtonRelease-1>", on_release)
    button.bind("<Button-1>", lambda e: command())

# Crează butoanele personalizate
create_custom_button("Programarea Pacientului", button1_action, 0.4)
create_custom_button("Diagnosticarea", button2_action, 0.6)

# Rulați aplicația
root.mainloop()
