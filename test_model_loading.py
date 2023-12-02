
"""
Code for question 3
"""
import os
import joblib
from sklearn.linear_model import LogisticRegression

def test_logistic_regression_model_loading():
    model_path = "models/final_exam"

    solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    rollno = "m22aie233"

    for solver in solvers:
        model_filename = f"{model_path}/{rollno}_lr_{solver}.joblib"
        assert os.path.exists(model_filename), f"Model file {model_filename} does not exist."

        model = joblib.load(model_filename)
        assert isinstance(model, LogisticRegression), f"Loaded model is not a Logistic Regression model."

def test_solver_name_in_model():
    model_path = "models/final_exam"

    solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    rollno = "m22aie233"

    for solver in solvers:
        model_filename = f"{model_path}/{rollno}_lr_{solver}.joblib"
        model = joblib.load(model_filename)
        
        assert model.get_params()['solver'] == solver, f"Solver in model {model_filename} does not match expected solver {solver}."
