import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load dataset
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=42)

# Define and train the production model (SVM), Hyperparam tuning with GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
svm = GridSearchCV(SVC(), param_grid, cv=5)
svm.fit(X_train, y_train)
prod_pred = svm.predict(X_test)
prod_acc = accuracy_score(y_test, prod_pred)

# Define and train the candidate model (Decision Tree), Hyperparam tuning with GridSearchCV
param_grid = {'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
dt = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
dt.fit(X_train, y_train)
cand_pred = dt.predict(X_test)
cand_acc = accuracy_score(y_test, cand_pred)

# Calculate confusion matrices
conf_matrix_prod_cand = confusion_matrix(prod_pred, cand_pred)

# Creating a 2x2 confusion matrix for samples predicted correctly in production but not in candidate.
true_positive = np.sum((prod_pred == y_test) & (cand_pred == y_test))
false_positive = np.sum((prod_pred != y_test) & (cand_pred == y_test))
false_negative = np.sum((prod_pred == y_test) & (cand_pred != y_test))
true_negative = np.sum((prod_pred != y_test) & (cand_pred != y_test))
conf_matrix_2x2 = np.array([[true_positive, false_positive], [false_negative, true_negative]])

# Calculate macro-average F1 score
f1_prod = f1_score(y_test, prod_pred, average='macro')
f1_cand = f1_score(y_test, cand_pred, average='macro')

#results
print("Production model's accuracy:", prod_acc)
print("Candidate model's accuracy:", cand_acc)
print("Confusion matrix between predictions of production and candidate models:\n", conf_matrix_prod_cand)
print("2x2 Confusion Matrix (Rows: Production, Columns: Candidate):\n", conf_matrix_2x2)
print("Production model macro-average F1 score:", f1_prod)
print("Candidate model macro-average F1 score:", f1_cand)
