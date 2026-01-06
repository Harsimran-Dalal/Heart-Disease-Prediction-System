# ❤️ Heart Disease Prediction System

An end-to-end **Machine Learning web application** that predicts the risk of heart disease based on clinical parameters.  
The system is trained using classical ML algorithms and deployed as an interactive web app using **Streamlit**.

---

## Project Overview

Cardiovascular diseases are among the leading causes of death worldwide. Early risk assessment can help in preventive care and timely medical intervention.

This project uses a **Random Forest classifier** to predict whether a patient is at **high risk or low risk of heart disease**, based on medical attributes such as age, cholesterol, blood pressure, chest pain type, etc.

The final model is deployed as a **user-friendly web application**.

---

## Features

- Machine Learning–based risk prediction  
- Clean & modern UI (Glassmorphism design)  
- Probability-based output instead of only Yes/No  
- Responsive layout (works on mobile & desktop)  
- Deployed on Streamlit Cloud  
- Production-ready model loading using `joblib`  

---

## Dataset

- **Source:** Kaggle – Heart Disease Dataset  
- **Target Variable:**  
  - `1` → High risk of heart disease  
  - `0` → Low risk of heart disease  

### Input Features
- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Resting ECG  
- Maximum Heart Rate  
- Exercise Induced Angina  
- ST Depression (oldpeak)  
- Slope of ST Segment  
- Number of Major Vessels  
- Thalassemia  

---

## Machine Learning Model

- **Algorithm Used:** Random Forest Classifier  
- **Why Random Forest?**
  - Handles non-linear relationships well
  - Robust to overfitting
  - Works effectively on tabular medical data

### Preprocessing Steps
- Feature scaling using `StandardScaler`
- Train–test split with stratification
- Model serialization using `joblib`

---

## Web Application

- Built using **Streamlit**
- Users can input medical parameters via dropdowns and sliders
- Outputs:
  - Risk classification (High / Low)
  - Probability score
  - Visual risk progress bar

---

## Installation & Running Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Harsimran-Dalal/heart-disease-prediction-system.git
cd heart-disease-prediction-system
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application
```bash
streamlit run app.py
```

## Disclaimer

This application is intended for educational purposes only.
It should not be used as a substitute for professional medical diagnosis.



