import numpy as np
import cv2
import os
import tensorflow as tf
from keras.models import load_model
from tkinter import *
import tkinter.messagebox
import PIL.Image
import PIL.ImageTk
from tkinter import filedialog
import csv
from transformers import pipeline  # For GPT-2 integration

# Load the GPT-2 model for text generation
generator = pipeline("text-generation", model="gpt2")

# Load the CSV file for cosmetics
file = open('cosmetics.csv')
csvreader = csv.reader(file)
header = next(csvreader)

DATADIR = "train"
CATEGORIES = os.listdir(DATADIR)

# Set up the GUI
root = Tk()
root.title("COSMETIC SUGGESTION")
root.state('zoomed')
root.configure(bg='#D3D3D3')
root.resizable(width=True, height=True)
value = StringVar()
panel = Label(root)
model = tf.keras.models.load_model("CNN.model")

def Camera():
    # Open the camera to capture an image
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        cv2.imwrite('main.jpg', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

def prepare(file):
    # Preprocess the image for CNN model
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.equalizeHist(img_array)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def generate_remedies(skin_type):
    """
    Generate remedies and suggestions based on skin type using GPT-2.
    """
    prompt = f"Suggest remedies and tips for someone with {skin_type} skin."
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

def detect(filename):
    """
    Detect skin type and display remedies using GPT-2.
    """
    prediction = model.predict(prepare(filename))
    prediction = list(prediction[0])
    skin_type = CATEGORIES[prediction.index(max(prediction))]
    i=int(prediction.index(max(prediction)))
    i=1
    file = open('cosmetics.csv')
    type(file)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    j=0
    for row in csvreader:
        print(row[3])
        if i==int(row[3]):
            x=header[0]+" : "+row[0]+"\n"+header[1]+" : "+row[1]+"\n"+header[2]+" : "+row[2]+"\n"
            tkinter.messagebox.showinfo("",x)
    value.set(skin_type)

    # Generate remedies using GPT-2
    remedies = generate_remedies(skin_type)

    # Display remedies in a message box
    tkinter.messagebox.showinfo("Suggestions", f"Skin Type: {skin_type}\n\nRemedies:\n{remedies}")

def ClickAction(event=None):
    """
    Handle the file selection and pass the file for detection.
    """
    filename = filedialog.askopenfilename()
    img = PIL.Image.open(filename)
    img = img.resize((250, 250))
    img = PIL.ImageTk.PhotoImage(img)
    global panel
    panel = Label(root, image=img)
    panel.image = img
    panel = panel.place(relx=0.435, rely=0.3)
    detect(filename)

# GUI Components
button = Button(root, text='CHOOSE FILE', font=(None, 18), activeforeground='red', bd=20, bg='cyan', relief=RAISED, height=3, width=20, command=ClickAction)
button.place(relx=0.40, rely=0.05)
result = Label(root, textvariable=value, font=(None, 20))
result.place(relx=0.465, rely=0.7)

# Start the GUI
root.mainloop()
