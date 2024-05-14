import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
from keras.models import load_model
import numpy as np

def load_image(image_path):
    try:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = Image.open(image_path).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        return data
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def make_prediction(image_path):
    if image_path:
        try:
            # Preprocess image
            img_array = load_image(image_path)
            if img_array is None:
                return

            # Load model
            model = load_model("simpsons_cnn.h5", compile=False)

            class_names = open("labels.txt", "r").readlines()
            # Make prediction
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)


            # Print the predicted class
            predicted_class = class_names[predicted_class_index]
            confidence_score = prediction[0][predicted_class_index]
            prediction_text.set(f"Predicted Class: {predicted_class}")


        except Exception as e:
            print(f"Error making prediction: {e}")
            prediction_text.set("Error: Could not make prediction.")
    else:
        prediction_text.set("Please select an image first.")

def select_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    image_path_label.config(text=f"Selected Image: {selected_image_path}")

# Create the main window
root = tk.Tk()
root.title("Image Prediction App")

# Create image path label
image_path_label = tk.Label(root, text="Selected Image: None")
image_path_label.pack()

# Create "Select Image" button
select_image_button = tk.Button(root, text="Select Image", command=select_image)
select_image_button.pack(pady=10)

# Create prediction text variable
prediction_text = tk.StringVar()
prediction_text.set("Make a prediction by selecting an image and clicking the button below.")

# Create prediction label
prediction_label = tk.Label(root, textvariable=prediction_text)
prediction_label.pack()

# Create "Make Prediction" button
make_prediction_button = tk.Button(root, text="Make Prediction", command=lambda: make_prediction(selected_image_path))
make_prediction_button.pack(pady=10)

# Run the main loop
root.mainloop()