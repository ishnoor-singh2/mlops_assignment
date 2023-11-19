from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np
import logging

# Load the pre-trained classifier

model_path = r"/Users/ishnoorsingh/Code/mlops_assignment/models/model.pkl"

# model_path = r"/Users/ishnoorsingh/Code/mlops_assignment/models/random_forest_digit_classifier.pkl"
clf = joblib.load(model_path)

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Endpoint for predicting a single digit from an image.
    """
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_data = preprocess(image_file)
        prediction = clf.predict([image_data])[0]
        app.logger.info(f'Predicted digit: {int(prediction)}')
        return jsonify({'predicted_digit': int(prediction)})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Error processing the image'}), 500

@app.route('/predict-compare', methods=['POST'])
def predict_compare():
    """
    Endpoint to compare two images and determine if they represent the same digit.
    """
    image_files = request.files.getlist('images')
    
    if len(image_files) != 2:
        return jsonify({'error': 'Need two images for comparison'}), 400

    try:
        image1_data = preprocess(image_files[0])
        image2_data = preprocess(image_files[1])

        # Make predictions
        prediction1 = clf.predict([image1_data])[0]
        prediction2 = clf.predict([image2_data])[0]

        is_same_digit = prediction1 == prediction2
        return jsonify({'are_same_digit': bool(is_same_digit)})
    except Exception as e:
        logging.error(f"Error during comparison: {str(e)}")
        return jsonify({'error': 'Error processing images'}), 500
    



def preprocess(image_file):
    """
    Preprocesses the image file to be in the format expected by the classifier.
    """
    with Image.open(image_file) as img:
        img = img.convert('L')

        # Ensure the image is resized to 8x8 pixels
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        
        image_array = np.array(img, dtype=np.float64).flatten()

        # Normalize only if your images are in the 0-255 range
        image_array *= (16 / 255) 

    return image_array

if __name__ == '__main__':
    app.run(debug=True)
