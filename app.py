import os
import numpy as np
import pandas as pd
import cv2
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the CNN model
model = load_model("CNN.model")

# Load GPT-2 generator
generator = pipeline("text-generation", model="gpt2")

# Load cosmetics data
cosmetics_data = pd.read_csv('cosmetics.csv')

# Categories for skin types
DATADIR = "train"
CATEGORIES = os.listdir(DATADIR)

def prepare(file):
    """
    Preprocess the image for the CNN model.
    """
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.equalizeHist(img_array)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def generate_remedies(skin_type):
    """
    Generate remedies and tips for a given skin type using GPT-2.
    """
    prompt = f"Suggest remedies and tips for someone with {skin_type} skin."
    result = generator(prompt, max_length=500, num_return_sequences=1)
    return result[0]['generated_text']

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main page for uploading an image.
    """
    if request.method == "POST":
        # Save uploaded file
        file = request.files['file']
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Predict skin type
        prediction = model.predict(prepare(file_path))
        prediction = list(prediction[0])
        skin_type = CATEGORIES[prediction.index(max(prediction))]
        print(skin_type)

        # Get cosmetics data for the detected skin type
        skin_type_index = prediction.index(max(prediction))
        print(skin_type_index)
        recommended_cosmetics = cosmetics_data[cosmetics_data['skintype'] == skin_type_index]
        print(recommended_cosmetics)

        

        # Generate remedies using GPT-2
        remedies = generate_remedies(skin_type)

        # Pass results to result page
        return render_template("result.html", skin_type=skin_type, remedies=remedies, cosmetics=recommended_cosmetics.to_dict(orient='records'))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
