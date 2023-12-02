# code for question 4

from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np

app = Flask(__name__)

# Function to load models
def load_model(model_type):
    model_path = {
        "svm": "/Users/ishnoorsingh/Code/mlops_assignment/models/final_exam/svm_model.joblib",
        "lr": "/Users/ishnoorsingh/Code/mlops_assignment/models/final_exam/m22aie233_lr_saga.joblib",
        "tree": "/Users/ishnoorsingh/Code/mlops_assignment/models/final_exam/random_forest_digit_classifier.pkl"
    }
    return joblib.load(model_path[model_type])

# Modified predict route to handle different models
@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    if model_type not in ['svm', 'lr', 'tree']:
        return jsonify({'error': 'Invalid model type'}), 400

    clf = load_model(model_type)

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
    app.run(debug=True,host = '0.0.0.0')
