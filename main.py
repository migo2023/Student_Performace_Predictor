import altair as alt
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st

dataset = pd.read_csv('training_data_student_perf.csv')
X_train = dataset[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y_train = dataset['Performance Index']

for i in range(0, X_train.shape[0]):
  if X_train.at[i, "Extracurricular Activities"] == "Yes":
    X_train.at[i, "Extracurricular Activities"] = 1
  elif X_train.at[i, "Extracurricular Activities"] == "No":
    X_train.at[i, "Extracurricular Activities"] = 0

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

num_iters = st.slider("Number of iterations for running gradient descent", 1, 100000, 10000)
sgdr = SGDRegressor(max_iter=num_iters)
sgdr.fit(X_norm, y_train)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_

test_dataset = pd.read_csv('test_data_student_perf.csv')
X_test = test_dataset[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
for i in range(0, X_test.shape[0]):
  if X_test.at[i, "Extracurricular Activities"] == "Yes":
    X_test.at[i, "Extracurricular Activities"] = 1
  elif X_test.at[i, "Extracurricular Activities"] == "No":
    X_test.at[i, "Extracurricular Activities"] = 0
X_test_norm = scaler.fit_transform(X_test)

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_test_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_test_norm, w_norm) + b_norm  

test_dataset.insert(6, "H-index(as predicted)", y_pred_sgd, True)

st.write(test_dataset.head())

