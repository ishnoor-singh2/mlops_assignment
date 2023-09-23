import matplotlib.pyplot as plt
from itertools import product
import utils
from skimage.transform import resize  # Import for resizing

# Load the digits dataset
data, labels = utils.load_digits_data()

### QUIZ ANSWERS
# 2.1 Print the number of total samples in the dataset
print(f"QUIZ ANSWER 2.1 ****Total samples in the dataset: {len(data)}")

# 2.2 Print the size of the images in the dataset
image_shape = data[0].reshape(int(len(data[0])**0.5), -1).shape
print(f"QUIZ ANSWER 2.2 ****Size of the images in the dataset: {image_shape}")

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
list_of_all_param_combination_dictionaries = [{'gamma': gamma, 'C': c} for gamma, c in product(gamma_ranges, C_ranges)]

# Define desired image sizes for resizing
image_sizes = [(4, 4), (6, 6), (8, 8)]

for img_size in image_sizes:
    # Resize images
    resized_data = [resize(image.reshape(8, 8), img_size).flatten() for image in data]
    
    # Split data using the function
    X_train, X_dev, X_test, y_train, y_dev, y_test = utils.split_train_dev_test(resized_data, labels, test_size=0.2, dev_size=0.1)
    
    # Hyperparameter tuning
    best_hparams, best_model, _ = utils.tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination_dictionaries)
    
    # Calculate accuracies
    train_acc, _, _ = utils.predict_and_eval(best_model, X_train, y_train)
    dev_acc, _, _ = utils.predict_and_eval(best_model, X_dev, y_dev)
    test_acc, _, _ = utils.predict_and_eval(best_model, X_test, y_test)
    
    # Print results for the resized images
    print(f"image size: {img_size[0]}x{img_size[1]} train_size: 0.7 dev_size: 0.1 test_size: 0.2 train_acc: {train_acc:.2f} dev_acc: {dev_acc:.2f} test_acc: {test_acc:.2f}")
    print(f"Best Hyperparameters: {best_hparams}")
