# main.py

import matplotlib.pyplot as plt
from itertools import product
import utils

# Load the digits dataset
data, labels = utils.load_digits_data()

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
list_of_all_param_combination_dictionaries = [{'gamma': gamma, 'C': c} for gamma, c in product(gamma_ranges, C_ranges)]

for test_size in [0.1, 0.2, 0.3]:
    for dev_size in [0.1, 0.2, 0.3]:
        # Split data using the function
        X_train, X_dev, X_test, y_train, y_dev, y_test = utils.split_train_dev_test(data, labels, test_size=test_size, dev_size=dev_size)
        
        # Hyperparameter tuning
        best_hparams, best_model, _ = utils.tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination_dictionaries)
        
        # Calculate accuracies
        train_acc, _, _ = utils.predict_and_eval(best_model, X_train, y_train)
        dev_acc, _, _ = utils.predict_and_eval(best_model, X_dev, y_dev)
        test_acc, _, _ = utils.predict_and_eval(best_model, X_test, y_test)
        
        print(f"test_size={test_size} dev_size={dev_size} train_size={1-test_size:.1f} train_acc={train_acc:.2f} dev_acc={dev_acc:.2f} test_acc={test_acc:.2f}")
        print(f"Best Hyperparameters: {best_hparams}")
