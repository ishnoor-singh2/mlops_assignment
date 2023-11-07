from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np


clf = joblib.load('digit_classifier.joblib')  

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    image_files = request.files.getlist('images')
    
    if len(image_files) != 2:
        return jsonify({'error': 'Need two images for comparison'}), 400

    image1_data = preprocess(image_files[0])
    image2_data = preprocess(image_files[1])

    # Make predictions
    prediction1 = clf.predict([image1_data])[0]
    prediction2 = clf.predict([image2_data])[0]

    is_same_digit = prediction1 == prediction2
    is_same_digit = bool(is_same_digit)

    return jsonify({'are_same_digit': is_same_digit})



def preprocess(image_file):
    """
    Preprocesses the image file to be in the format expected by the classifier.

    """
    with Image.open(image_file) as img:
        img = img.convert('L')
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        image_array = np.array(img, dtype=np.float64)
        image_array = image_array.flatten()
        image_array *= (16 / 255) 

    return image_array


if __name__ == '__main__':
    app.run(debug=True)
