import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0f1117;
}
h1, h2, h3 {
    color: white;
    text-align: center;
}
label {
    font-weight: 600;
}
.stButton>button {
    width: 100%;
    padding: 0.6rem;
    font-size: 1.1rem;
    border-radius: 10px;
}
.card {
    background-color: #1c1f26;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-based clinical risk assessment</p>", unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)

    slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.markdown("##")
if st.button("🔍 Predict Heart Disease Risk"):
    input_data = np.array([
        age,
        1 if sex == "Male" else 0,
        cp,
        trestbps,
        chol,
        1 if fbs == "Yes" else 0,
        restecg,
        thalach,
        1 if exang == "Yes" else 0,
        oldpeak,
        slope,
        ca,
        thal
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected**\n\nProbability: **{probability:.2%}**")
    else:
        st.success(f"✅ **Low Risk Detected**\n\nProbability: **{probability:.2%}**")

# ---------------- FOOTER ----------------
st.markdown("""
<p style='text-align:center; font-size:0.9rem; color:gray;'>
This tool is for educational purposes only and not a medical diagnosis.
</p>
""", unsafe_allow_html=True)
