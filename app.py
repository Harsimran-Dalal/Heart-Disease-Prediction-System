import streamlit as st
import numpy as np
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model()

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #020617, #020617);
    color: white;
}

html, body, [class*="css"] {
    color: white;
}

.title {
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #f43f5e, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 0 40px rgba(0,0,0,0.4);
}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    padding: 0.8rem;
    font-size: 1.05rem;
    font-weight: 600;
    background: linear-gradient(90deg, #ec4899, #f43f5e);
    color: white;
    border: none;
}

.risk-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    padding: 20px;
    border-radius: 16px;
    color: #fecaca;
}

.risk-low {
    background: linear-gradient(135deg, #064e3b, #065f46);
    padding: 20px;
    border-radius: 16px;
    color: #bbf7d0;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 700;
    margin-top: 10px;
}

.footer {
    text-align: center;
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 25px;
}

@media (max-width: 768px) {
    .card {
        padding: 20px;
    }
    .title {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("<div class='title'>💗 Heart Disease Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-based clinical risk assessment</div>", unsafe_allow_html=True)

# ================= INPUT CARD =================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
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

# ================= PREDICTION =================
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

    risk_percent = probability * 100

    st.markdown("---")
    st.markdown("### 🧠 Prediction Result")
    st.progress(min(int(risk_percent), 100))

    if prediction == 1:
        st.markdown(
            f"""
            <div class="risk-high">
                <h3>⚠️ High Risk Detected</h3>
                <span class="badge">
                    Probability: {risk_percent:.2f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="risk-low">
                <h3>✅ Low Risk Detected</h3>
                <span class="badge">
                    Probability: {risk_percent:.2f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ================= FOOTER =================
st.markdown(
    "<div class='footer'>For educational purposes only • Not a medical diagnosis</div>",
    unsafe_allow_html=True
)
