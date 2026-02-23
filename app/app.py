import os
import joblib # pyright: ignore[reportMissingImports]
import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd

# ----------------------------
# Load model and feature order
# ----------------------------

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build correct paths
MODEL_PATH = os.path.join(BASE_DIR, "../models/flu_risk_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "../models/feature_order.pkl")

# Load files
model = joblib.load(MODEL_PATH)
feature_order = joblib.load(FEATURE_PATH)

# ----------------------------
# App Title
# ----------------------------
st.title("Flu Risk Prediction App")
st.write("Enter your demographics and symptoms to estimate your risk of flu.")

# ----------------------------
# Demographics Input
# ----------------------------
st.header("Patient Information")

age_group = st.selectbox(
    "Age Group", 
    ["0-18", "19-35", "36-50", "51-65", "66+"], 
    index=1
)

sex = st.selectbox(
    "Sex", 
    ["Male", "Female"]
)

# Map to encoded values (same as training)
age_map = {"0-18":0, "19-35":1, "36-50":2, "51-65":3, "66+":4}
sex_map = {"Male":0, "Female":1}

age_encoded = age_map[age_group]
sex_encoded = sex_map[sex]

# ----------------------------
# Symptoms Input
# ----------------------------
st.header("Symptoms (Check if Yes)")

# Original symptom columns from your dataset
symptoms = [
    "COUGH", "FEVER", "VOMITING", "DIARRHEA", "TIREDNESS", "MUSCLE_ACHES",
    "DIFFICULTY_BREATHING", "SHORTNESS_OF_BREATH", "SORE_THROAT",
    "RUNNY_NOSE", "STUFFY_NOSE", "SNEEZING", "LOSS_OF_TASTE", "LOSS_OF_SMELL",
    "NAUSEA", "ITCHY_NOSE", "ITCHY_EYES", "ITCHY_MOUTH", "ITCHY_INNER_EAR", "PINK_EYE"
]

# Collect symptoms from checkboxes
input_data_dict = {}
for symptom in symptoms:
    input_data_dict[symptom] = [st.checkbox(symptom)]

# ----------------------------
# Create DataFrame
# ----------------------------
input_df = pd.DataFrame(input_data_dict)

# Add encoded demographics
input_df["AGE_GROUP_ENCODED"] = age_encoded
input_df["SEX_ENCODED"] = sex_encoded

# ----------------------------
# Compute Symptom Counts (Feature Engineering)
# ----------------------------
# Total symptom count
input_df["SYMPTOM_COUNT"] = input_df[symptoms].sum(axis=1)

# Grouped counts (example based on your earlier engineering)
resp_symptoms = ["COUGH","DIFFICULTY_BREATHING","SHORTNESS_OF_BREATH","SORE_THROAT","RUNNY_NOSE"]
gi_symptoms = ["VOMITING","DIARRHEA","NAUSEA"]
allergy_symptoms = ["ITCHY_NOSE","ITCHY_EYES","ITCHY_MOUTH","ITCHY_INNER_EAR","PINK_EYE"]

input_df["RESP_SYMPTOM_COUNT"] = input_df[resp_symptoms].sum(axis=1)
input_df["GI_SYMPTOM_COUNT"] = input_df[gi_symptoms].sum(axis=1)
input_df["ALLERGY_SYMPTOM_COUNT"] = input_df[allergy_symptoms].sum(axis=1)

# ----------------------------
# Reorder columns exactly as training
# ----------------------------
input_df = input_df[feature_order]

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Flu Risk"):
    risk_prob = model.predict_proba(input_df)[0][1]  # probability of flu
    st.subheader("Flu Risk Probability")
    st.write(f"{risk_prob*100:.2f}%")
    
    if risk_prob > 0.5:
        st.warning("High risk of flu. Consider consulting a healthcare professional.")
    else:
        st.success("Low risk of flu. Stay safe!")
