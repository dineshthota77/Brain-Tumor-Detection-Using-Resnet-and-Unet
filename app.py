from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory for uploaded files

# Load your trained model
try:
    model = load_model('best_model_final.keras')  # Ensure this path is correct
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class labels for prediction output
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Home route to upload files
@app.route('/')
def index():
    return render_template('index.html', result=None)

# Prediction route to handle uploaded files
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="No file part in the request")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No file selected")

    # Save the uploaded file
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File saved to {file_path}")

        try:
            # Load and preprocess the image
            img = Image.open(file_path).convert('RGB')  # Ensure image is in RGB format
            img = img.resize((224, 224))  # Adjust size based on your model's input
            img = np.array(img) / 255.0  # Normalize the image
            
            # Check the shape of the image and expand dimensions if necessary
            if img.ndim == 3:  # If the image has shape (224, 224, 3)
                img = np.expand_dims(img, axis=0)  # Add batch dimension
            elif img.ndim == 2:  # If the image has shape (224, 224)
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                img = np.repeat(img, 3, axis=-1)  # Convert grayscale to RGB
                img = np.expand_dims(img, axis=0)  # Add batch dimension
            else:
                raise ValueError("Unsupported image shape: {}".format(img.shape))

            # Perform prediction
            prediction = model.predict(img)
            print(f"Prediction raw output: {prediction}")
            
            # Get the class with the highest probability
            predicted_class = class_labels[np.argmax(prediction)]
            print(f"Predicted Class: {predicted_class}")

            # Remove the file after prediction
            if os.path.exists(file_path):
                os.remove(file_path)
                print("File removed after prediction")

            # Return the result in the template
            return render_template('index.html', result=f"Predicted Class: {predicted_class}")
        
        except Exception as e:
            # Log and return any error that occurs during prediction
            print(f"Error occurred during prediction: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template('index.html', result=f"Error occurred during prediction: {str(e)}")

    return render_template('index.html', result="Error occurred!")

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for development
