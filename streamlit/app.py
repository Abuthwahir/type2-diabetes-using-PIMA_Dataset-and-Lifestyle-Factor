import streamlit as st          # For building the web UI
import pandas as pd             # For data handling
import joblib                   # For loading trained ML models
import os                       # For file path handling
import matplotlib.pyplot as plt # (Optional) For plots if needed later
import matplotlib.colors as mcolors

# -------------------------------------------------
# Page configuration (title, icon, layout)
# -------------------------------------------------
st.set_page_config(
    page_title="Type-2 Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------------------------
# Load trained model and feature names
# Using cache so model loads only once (faster app)
# -------------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Current file directory
    pipeline = joblib.load(os.path.join(base_dir, "clinical_diabetes_pipeline.pkl"))  # ML pipeline
    feature_names = joblib.load(os.path.join(base_dir, "feature_names.pkl"))          # Feature order
    return pipeline, feature_names

pipeline, feature_names = load_model()

# -------------------------------------------------
# Clinical definitions (just for display/help text)
# -------------------------------------------------
CLINICAL_DEFINITIONS = {
    "Glucose": "Fasting plasma glucose (mg/dL)",
    "BloodPressure": "Diastolic blood pressure (mmHg)",
    "BMI": "Body Mass Index (kg/m2)",
    "Age": "Age in years"
}

# -------------------------------------------------
# App Title and Subtitle
# -------------------------------------------------
st.title("🩺 Type-2 Diabetes Risk Assessment")
st.markdown(
    "<i>Interpretable ML-based Type-2 diabetes risk assessment (not a diagnosis)</i>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# Sidebar: User Inputs
# -------------------------------------------------
with st.sidebar:
    st.header("Patient Information")

    # -------------------------
    # Basic demographics
    # -------------------------
    st.subheader("Basic Demographics")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    # Pregnancies only applicable for females
    pregnancies = 0
    if gender == "Female":
        pregnancies = st.number_input(
            "Number of pregnancies",
            min_value=0,
            max_value=20,
            value=0
        )

    # -------------------------
    # BMI input or calculation
    # -------------------------
    st.divider()
    st.subheader("BMI Calculator")

    bmi_method = st.radio(
        "BMI input method",
        ["Enter BMI directly", "Calculate from height/weight"]
    )

    bmi_value = 26.0
    if bmi_method == "Calculate from height/weight":
        height_cm = st.number_input("Height (cm)", 100, 250, 170)
        weight_kg = st.number_input("Weight (kg)", 30, 200, 75)
        bmi_value = weight_kg / ((height_cm / 100) ** 2)  # BMI formula
        st.success(f"BMI = {bmi_value:.1f}")
    else:
        bmi_value = st.slider("BMI (kg/m2)", 15.0, 50.0, 26.0)

    # -------------------------
    # Core clinical inputs
    # -------------------------
    st.divider()
    glucose = st.slider("Glucose (mg/dL)", 50, 300, 120)
    bp = st.slider("Diastolic BP (mmHg)", 30, 120, 70)
    age = st.slider("Age (years)", 20, 85, 35)

    # -------------------------
    # Lifestyle factors (used to adjust risk)
    # -------------------------
    st.divider()
    st.subheader("Lifestyle Factors")

    smoking = st.radio("Smoking", ["No", "Yes"], horizontal=True)
    activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])
    diet = st.selectbox("Diet type", ["Rice-based", "Mixed", "High-protein", "Junk food"])
    meals = st.selectbox("Meals per day", ["1-2", "3", "4+"])
    family = st.radio("Family history of diabetes", ["No", "Yes"], horizontal=True)
    cardio = st.radio("Heart disease / stroke history", ["No", "Yes"], horizontal=True)

    # -------------------------
    # Optional clinical inputs
    # -------------------------
    st.divider()
    st.subheader("Optional Clinical Inputs")

    use_insulin = st.radio("Include Insulin?", ["No", "Yes"], horizontal=True)
    use_dpf = st.radio("Include Genetic Risk?", ["No", "Yes"], horizontal=True)

    # Default values if not provided
    insulin_value = 80
    dpf_value = 0.5

    if use_insulin == "Yes":
        insulin_value = st.slider("Insulin (µU/mL)", 0, 300, 80)

    if use_dpf == "Yes":
        dpf_value = st.slider(
            "Diabetes Pedigree Function",
            0.05, 2.5, 0.5, step=0.01
        )

    st.info(
        "When optional clinical values are unavailable, clinically reasonable defaults are used."
    )

# -------------------------------------------------
# Prepare input data in the exact format expected by the model
# -------------------------------------------------
def get_input_df():
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": 20,          # Hidden default value
        "Insulin": insulin_value,
        "BMI": bmi_value,
        "DiabetesPedigreeFunction": dpf_value,
        "Age": age
    }
    # Create DataFrame with correct column order
    return pd.DataFrame([data], columns=feature_names)

# -------------------------------------------------
# Lifestyle-based risk modifier (simple, explainable rules)
# This adjusts the ML probability to reflect lifestyle factors
# -------------------------------------------------
def apply_lifestyle_modifier(prob):
    delta = 0.0
    reasons = []

    if smoking == "Yes":
        delta += 0.03
        reasons.append("Smoking increases insulin resistance")

    if activity == "Low":
        delta += 0.03
        reasons.append("Low physical activity increases risk")
    elif activity == "High":
        delta -= 0.02
        reasons.append("High physical activity reduces risk")

    if diet == "Junk food":
        delta += 0.03
        reasons.append("High junk food intake increases risk")
    elif diet == "Rice-based":
        delta += 0.01
        reasons.append("High refined carbohydrate intake increases risk")

    if meals == "4+":
        delta += 0.02
        reasons.append("Frequent meals increase glucose load")

    if family == "Yes":
        delta += 0.04
        reasons.append("Family history increases susceptibility")

    if cardio == "Yes":
        delta += 0.04
        reasons.append("Cardiovascular disease is linked to metabolic risk")

    # Ensure probability stays between 0 and 0.99
    final_prob = min(max(prob + delta, 0), 0.99)
    return final_prob, reasons, prob

# -------------------------------------------------
# Prediction button logic
# -------------------------------------------------
if st.button("Assess Diabetes Risk", use_container_width=True):
    X = get_input_df()                              # Prepare input
    base_prob = pipeline.predict_proba(X)[0][1]     # Get model probability
    final_prob, reasons, raw_prob = apply_lifestyle_modifier(base_prob)

    st.subheader("Risk Result")

    # Assign risk category based on probability
    if final_prob < 0.2:
        label, color = "Very Low Risk", "#2ecc71"
    elif final_prob < 0.4:
        label, color = "Low Risk", "#27ae60"
    elif final_prob < 0.6:
        label, color = "Moderate Risk", "#f39c12"
    elif final_prob < 0.8:
        label, color = "High Risk", "#e74c3c"
    else:
        label, color = "Very High Risk", "#c0392b"

    # Display result nicely
    st.markdown(
        f"""
        <div style="padding:20px;border-left:6px solid {color};background:#f8f9fa">
        <h2 style="color:{color}">{label}</h2>
        <h3>{final_prob:.1%} probability</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show both raw model output and adjusted risk
    st.caption(
        f"Model probability: {raw_prob:.1%} | Lifestyle-adjusted: {final_prob:.1%}"
    )

    # Explain why risk was adjusted
    st.subheader("Why this risk level")
    if reasons:
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.markdown("- No lifestyle-based adjustments applied")