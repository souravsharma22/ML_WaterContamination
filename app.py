from flask import Flask, request, render_template, url_for
import os
import pickle
from PIL import Image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = pickle.load(open('Models/tunnedModel.pkl', 'rb'))

# Categories for prediction
categories = ['algae', 'clean', 'oil_spill', 'plastic']

@app.route('/', methods=['GET', 'POST'])
def home():
    message = ""
    image_url = None
    prediction = None
    predicted_category = None

    if request.method == 'POST':
        if 'image' not in request.files:
            message = "Chose File"
        else:
            file = request.files['image']
            if file.filename == '':
                message = "No selected file"
            elif file:
                # Save the uploaded file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                image_url = url_for('static', filename=f'uploads/{file.filename}')
                message = "Image uploaded successfully!"

                # Preprocess the image
                image = Image.open(filepath)
                image = image.resize((150, 150))  # Resize to 150x150
                image = np.array(image)  # Convert image to NumPy array

                # If the image is RGB, convert to grayscale or keep it as RGB based on model requirement
                if image.shape[-1] == 3:  # If RGB
                    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale (optional)

                # Reshape image for prediction
                image = image.reshape(1, -1)  # Flatten the image for SVC input (150*150,)
                
                # Predict using the model
                prediction = model.predict(image)
                predicted_category = categories[prediction[0]]  # Get the category from the prediction index

                
    return render_template('index.html', message=message, image_url=image_url, prediction=prediction, predicted_category=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)
