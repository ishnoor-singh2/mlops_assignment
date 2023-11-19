"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

# Changing the name from code.py to code_mlops_assignment.py to avoid naming conflict 
between the Python Standard Library's "code" module and your file named "code.py".
"""


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pickle

def split_train_dev_test(X, y, test_size=0.5, dev_size=0.25):
    """
    Splits data into train, dev, and test subsets.
    Return : Train , test and dev sets
    """
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=dev_size, shuffle=True, stratify=y_train_dev)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def predict_and_eval(model, X_test, y_test):
    """
    Predict using the model , evaluate the results and print report.

    return: predictions
    """
    predicted = model.predict(X_test)
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    return predicted


# Load the digits dataset
digits = datasets.load_digits()

# Flatten the images for processing
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data using the function
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(data, digits.target, test_size=0.2, dev_size=0.1)

# Create and train a RandomForest classifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
clf.fit(X_train, y_train)

print(f"Best parameters: {clf.best_params_}")

# Predict and evaluate on dev data
predicted_dev = predict_and_eval(clf, X_dev, y_dev)

# Predict and evaluate on test data
predicted_test = predict_and_eval(clf, X_test, y_test)

# Display the confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

# Save the model
with open("models/random_forest_digit_classifier.pkl", "wb") as f:
    pickle.dump(clf.best_estimator_, f)