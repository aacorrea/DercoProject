# https://www.section.io/engineering-education/deploying-machine-learning-models-using-flask/

import pandas as pd  # To manage data as data frames
import numpy as np  # To manipulate data as arrays
import joblib
from sklearn.linear_model import LogisticRegression  # Classification model
from xgboost.sklearn import XGBClassifier

# Dictionary containing the mapping
cluster_mappings = {0: 'Segmento A', 1: 'Segmento B'}

logreg = joblib.load('xgb_model_derco.sav')

# Function for classification based on inputs
def classify(a, b, c, d, e, f, g):
    arr = np.array([a, b, c, d, e, f, g])  # Convert to numpy array
    arr = arr.astype(np.int)  # Change the data type to int
    query = arr.reshape(1, -1)  # Reshape the array
    prediction = cluster_mappings[logreg.predict(query)[0]]  # Retrieve from dictionary
    return prediction  # Return the prediction
