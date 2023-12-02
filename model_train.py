"""
================================
Recognizing hand-written digits
================================

Code For Final Exam Of MLOPS , incorporates chnages
question 1) , question 2) solved here
"""


# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers, and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier

#immport for question1
from sklearn.preprocessing import Normalizer 
#imports for question3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib

rollno = "m22aie233"


def split_train_dev_test(X, y, test_size=0.5, dev_size=0.25):
    """
    Splits data into train, dev, and test subsets.
    Return : Train , test and dev sets
    """
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=dev_size, shuffle=True, stratify=y_train_dev)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def predict_and_eval(model, X_test, y_test, model_name):
    predicted = model.predict(X_test)
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    # Save the model
    path_to_save = r"/Users/ishnoorsingh/Code/mlops_assignment/models/final_exam"
    joblib.dump(model, f"{path_to_save}/{model_name}.joblib")
    print(f"Model saved as {model_name}.joblib")
    return predicted

def train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test, solvers):
    rollno = "m22aie233"

    for solver in solvers:
        model = LogisticRegression(solver=solver, max_iter=1000)
        model.fit(X_train, y_train)
        print(f"Evaluating model with solver '{solver}'")
        predict_and_eval(model, X_test, y_test, f"{rollno}_lr_{solver}")  # Add model_name here

        # Save the model
        model_save_path = r"/Users/ishnoorsingh/Code/mlops_assignment/models/final_exam"
        model_filename = f"{model_save_path}/{rollno}_lr_{solver}.joblib"
        joblib.dump(model, model_filename)
        print(f"Model saved as {model_filename}")

        # Bonus: Report mean and std of performance across 5 CV
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Mean CV Score for '{solver}': {cv_scores.mean()}")
        print(f"Standard Deviation CV Score for '{solver}': {cv_scores.std()}\n")


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

# Apply unit normalization , for question 1)
normalizer = Normalizer().fit(data)
data_normalized = normalizer.transform(data)

# Split data using the function
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(data_normalized, digits.target, test_size=0.2, dev_size=0.1)


#For question2
solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test, solvers)


# # SVM Classifier
# svm_clf = svm.SVC(gamma=0.001)
# svm_clf.fit(X_train, y_train)
# predict_and_eval(svm_clf, X_test, y_test, "svm_model")

# # RF Classifier
# rf_clf = RandomForestClassifier()
# rf_clf.fit(X_train, y_train)
# predict_and_eval(rf_clf, X_test, y_test, "rf_model")
