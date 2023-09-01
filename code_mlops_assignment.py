"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

# Changing the name from code.py to code_mlops_assignment.py to avoid naming conflict 
between the Python Standard Library's "code" module and your file named "code.py".
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers, and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def split_train_dev_test(X, y, test_size=0.5, dev_size=0.25):
    """
    Splits data into train, dev, and test subsets.

    return: train , dev and test set for both features and labels
    """
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=dev_size, shuffle=True, stratify=y_train_dev)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def predict_and_eval(model, X_test, y_test):
    """
    Predict using the model and evaluate the results, prints the classification report

    return: predictions from the model.predict()
    """
    predicted = model.predict(X_test)
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    return predicted

# Load the digits dataset
digits = datasets.load_digits()

# Visualize the first 4 images
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Flatten the images for processing
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data using the function
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(data, digits.target, test_size=0.2, dev_size=0.1)

# Create and train a support vector classifier
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)

# Predict and evaluate on dev data
predicted_dev = predict_and_eval(clf, X_dev, y_dev)

# Predict and evaluate on test data
predicted_test = predict_and_eval(clf, X_test, y_test)

# Visualize the first 4 test samples with their predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# Display the confusion matrix
disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

# Additional step to rebuild classification report from confusion matrix
y_true = []
y_pred = []
cm = disp.confusion_matrix
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
