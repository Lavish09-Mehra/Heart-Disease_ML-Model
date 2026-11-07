import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load the artifacts that actually exist in this folder.
model = joblib.load(BASE_DIR / "LR_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
columns = list(joblib.load(BASE_DIR / "columns.pkl"))          # full 18-column schema
scale_cols = list(scaler.feature_names_in_)                     # 5 numeric columns
model_features = list(model.feature_names_in_)                  # 15 features used by the model

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")
st.title("Heart Disease Risk Predictor")
st.caption("Logistic Regression model · educational use only")

# ---- Inputs -------------------------------------------------------------
age = st.slider("Age", 18, 100, 54)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 130)
cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 1000, 240)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 140)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
oldpeak = st.slider("Oldpeak (ST Depression)", -3.0, 6.5, 1.0, 0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


def build_features() -> pd.DataFrame:
    """Rebuild exactly the 15 features the model was trained on."""
    row = {col: 0 for col in columns}
    row.update({
        "Age": float(age),
        "RestingBP": float(resting_bp),
        "Cholesterol": float(cholesterol),
        "FastingBS": int(fasting_bs),
        "MaxHR": float(max_hr),
        "Oldpeak": float(oldpeak),
        "is_male": 1 if sex == "M" else 0,
        "is_ExerciseAngina": 1 if exercise_angina == "Y" else 0,
    })
    row[f"ChestPainType_{chest_pain}"] = 1
    row[f"RestingECG_{resting_ecg}"] = 1
    row[f"ST_Slope_{st_slope}"] = 1

    frame = pd.DataFrame([row], columns=columns)
    # scale only the 5 numeric columns, exactly like training
    frame[scale_cols] = scaler.transform(frame[scale_cols])
    # keep only the features the model uses
    return frame[model_features]


# ---- Predict ------------------------------------------------------------
if st.button("Predict", type="primary"):
    features = build_features()
    risk = float(model.predict_proba(features)[0, 1])
    prediction = int(risk >= 0.5)

    st.metric("Estimated risk", f"{risk * 100:.1f}%")
    st.progress(min(max(risk, 0.0), 1.0))

    if prediction == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")

    st.caption("This is not medical advice. Consult a doctor for a real diagnosis.")
