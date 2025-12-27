import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model()

st.title("Heart Disease Prediction App")

inputs = []
labels = [
    "Age", "Sex (1=Male, 0=Female)", "Chest Pain Type",
    "Resting BP", "Cholesterol", "Fasting Blood Sugar",
    "Rest ECG", "Max Heart Rate", "Exercise Induced Angina",
    "Oldpeak", "Slope", "CA", "Thal"
]

for label in labels:
    inputs.append(st.number_input(label))

if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
