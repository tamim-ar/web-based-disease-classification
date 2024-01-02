from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

# Load the trained model
model = tf.keras.models.load_model('model/mobilenetv2.h5')

# Configuration parameters
IMG_DIM = 256  # Update with the correct image dimensions used during training
LABELS = ["Dot", "Phytopthora", "Red Rust", "Rust", "Scab", "Styler End Root"]  # Update with your class labels
CONFIDENCE_THRESHOLD = 0.5  # Adjust the threshold as needed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        img = Image.open(file.stream)

        # Resize the image to match the model's expected input shape
        img = img.resize((IMG_DIM, IMG_DIM))

        # Convert image to a NumPy array
        img_array = np.array(img) / 255.0
        img_array = np.reshape(img_array, (1, IMG_DIM, IMG_DIM, 3))  # Assuming 3 channels, adjust if needed

        # Make predictions
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]

        # Convert confidence to percentage
        confidence_percentage = int(confidence * 100)

        # Check if the model is confident in any class
        if confidence_percentage >= CONFIDENCE_THRESHOLD * 100:
            # Get the class label from the index
            class_label = LABELS[class_index]
        else:
            # If confidence is below the threshold, consider it as "Not Classified"
            class_label = "Not Classified"

        return render_template('index.html', prediction=class_label, confidence=confidence_percentage)

if __name__ == '__main__':
    app.run()
