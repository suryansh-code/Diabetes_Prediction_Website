import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the SVM model
svm_model = pickle.load(open('svm_model.pkl', 'rb'))

# Function to standardize features
def std_scalar(df):
    std_X = StandardScaler()
    x = pd.DataFrame(std_X.fit_transform(df))
    return x

# Streamlit app
st.title("Diabetes Prediction")

# Create form for user input
with st.form("diabetes_form"):
    Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1, value=0)
    Glucose = st.number_input('Glucose', min_value=0, max_value=300, step=1, value=0)
    BloodPressure = st.number_input('BloodPressure', min_value=0, max_value=200, step=1, value=0)
    SalivaryAmylase = st.number_input('SalivaryAmylase', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
    Insulin = st.number_input('Insulin', min_value=0.0, max_value=1000.0, step=0.1, value=0.0)
    BMI = st.number_input('BMI', min_value=0.0, max_value=100.0, step=0.1, value=0.0)
    DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, step=0.01, value=0.0)
    Age = st.number_input('Age', min_value=0, max_value=120, step=1, value=0)
    
    submit_button = st.form_submit_button(label='Predict')

# Predict and display results
if submit_button:
    features = [Pregnancies, Glucose, BloodPressure, SalivaryAmylase, Insulin, BMI, DiabetesPedigreeFunction, Age]
    final_features = [np.array(features)]
    
    # Standardize the features
    feature_transform = std_scalar(final_features)
    
    # Make prediction
    prediction = svm_model.predict(feature_transform)
    result = "You Are Diabetic" if prediction == 1 else "You Are Non-Diabetic"
    
    # Display the result
    st.subheader("Prediction Result")
    st.write(result)
    
    # Display input data
    st.subheader("Input Data")
    st.write(f"Pregnancies: {Pregnancies}")
    st.write(f"Glucose: {Glucose}")
    st.write(f"BloodPressure: {BloodPressure}")
    st.write(f"SalivaryAmylase: {SalivaryAmylase}")
    st.write(f"Insulin: {Insulin}")
    st.write(f"BMI: {BMI}")
    st.write(f"DiabetesPedigreeFunction: {DiabetesPedigreeFunction}")
    st.write(f"Age: {Age}")
