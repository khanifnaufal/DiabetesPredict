import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="DiaPredict | AI Diabetes Assessment",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #1e88e5;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# Header
st.title("🏥 DiaPredict")
st.subheader("AI-Powered Diabetes Risk Assessment")
st.write("Enter the patient's medical metrics below to receive a real-time risk evaluation.")

# Form
with st.form("prediction_form"):
    st.write("### Patient Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, value=20)
        
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, value=80)
        bmi = st.number_input("BMI", min_value=0.0, format="%.1f", value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", value=0.471)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    submit = st.form_submit_button("Analyze Risk Profile")

# Prediction logic
if submit:
    # Prepare data for prediction
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    st.divider()
    
    # Show Results
    if prediction == 1:
        st.error("### ⚠️ High Risk Detected")
        st.metric("Probability Score", f"{round(probability * 100, 2)}%")
        st.warning("**Recommendation:** Our analysis suggests a high risk of diabetes. We strongly recommend consulting a healthcare professional for a detailed medical evaluation.")
    else:
        st.success("### ✅ Low Risk Detected")
        st.metric("Probability Score", f"{round(probability * 100, 2)}%")
        st.info("**Recommendation:** Our analysis suggests a low risk. maintain a healthy lifestyle with a balanced diet and regular physical activity to keep your risk levels low.")

# Footer
st.markdown("---")
st.caption("© 2026 DiaPredict Medical Intelligence Analytics. For educational purposes only. Not a substitute for professional medical advice.")
