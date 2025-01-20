import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

# Load the trained model
loaded_model = load_model('saved_model/reptile_classifier.h5')

# Class labels
class_labels = {0: 'lizards', 1: 'snakes', 2: 'turtles'}

# Function to preprocess and predict
def predict_image(file_path):
    # Preprocess the image
    img = load_img(file_path, target_size=(150, 150))  # Adjust size as per your model
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = loaded_model.predict(img_array)
    class_idx = np.argmax(predictions)
    return class_labels[class_idx]

# Function for file upload
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
        title="Choose an image"
    )
    if file_path:
        # Predict and show the result
        result = predict_image(file_path)
        result_label.config(text=f"Predicted class: {result}")
    else:
        result_label.config(text="No file selected.")

# Create a GUI window
root = tk.Tk()
root.title("Reptile Classifier")

# Add upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image, padx=20, pady=10)
upload_button.pack(pady=20)

# Add result label
result_label = tk.Label(root, text="Prediction will appear here.", font=("Arial", 14))
result_label.pack(pady=20)

# Run the GUI loop
root.mainloop()
