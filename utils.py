# utils.py

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_digits_data():
    """
    Load and return the digits dataset.
    """
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

def split_train_dev_test(X, y, test_size=0.5, dev_size=0.25):
    """
    Splits data into train, dev, and test subsets.
    Return: Train, test, and dev sets
    """
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=dev_size, shuffle=True, stratify=y_train_dev)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def predict_and_eval(model, X_test, y_test):
    """
    Predict using the model, evaluate the results and return metrics.

    return: accuracy, classification report, confusion matrix
    """
    predicted = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    classification_report = metrics.classification_report(y_test, predicted)
    confusion_matrix = metrics.confusion_matrix(y_test, predicted)
    return accuracy, classification_report, confusion_matrix

def tune_hparams(X_train, Y_train, x_dev, y_dev, list_of_all_param_combination_dictionaries):
    """
    Tune hyperparameters using the provided combinations.

    return: best_hparams, best_model, best_accuracy
    """
    best_hparams = None
    best_model = None
    best_accuracy = 0

    for param_combination in list_of_all_param_combination_dictionaries:
        # Create a model with the current parameter combination
        model = svm.SVC(**param_combination)

        # Train the model
        model.fit(X_train, Y_train)

        # Validate the model
        accuracy = model.score(x_dev, y_dev)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_hparams = param_combination

    return best_hparams, best_model, best_accuracy
