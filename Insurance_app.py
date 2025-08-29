import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc

# -------------------------------
# Load best model from MLflow
# -------------------------------
MODEL_NAME = "InsurancePremiumModel"
MODEL_STAGE = "Production"  # or "Staging"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

st.set_page_config(page_title="Insurance Premium Predictor", layout="wide")

st.title("💰 Insurance Premium Prediction App")
st.write("Fill in the details below to predict the insurance premium.")

# -------------------------------
# Input Form
# -------------------------------
with st.form("premium_form"):
    st.subheader("Customer Information")

    # Numeric inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income", min_value=1000, max_value=1000000, value=50000)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
    health_score = st.slider("Health Score", 0, 100, 75)
    prev_claims = st.number_input("Previous Claims", min_value=0, max_value=20, value=0)
    vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    ins_duration = st.number_input("Insurance Duration (years)", min_value=0, max_value=50, value=5)

    st.subheader("Categorical Features")

    gender = st.radio("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
    smoking = st.radio("Smoking Status", ["Yes", "No"])
    exercise = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment"])

    submit = st.form_submit_button("Predict Premium")

# -------------------------------
# Preprocess input to match training features
# -------------------------------
if submit:
    # Numeric features
    input_data = {
        "Age": age,
        "Annual Income": income,
        "Number of Dependents": dependents,
        "Health Score": health_score,
        "Previous Claims": prev_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": ins_duration,
    }

    # One-hot categorical features (set selected=1, others=0)
    cats = {
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Marital Status_Single": 1 if marital_status == "Single" else 0,
        "Marital Status_Married": 1 if marital_status == "Married" else 0,
        "Marital Status_Divorced": 1 if marital_status == "Divorced" else 0,
        "Education Level_High School": 1 if education == "High School" else 0,
        "Education Level_Bachelor's": 1 if education == "Bachelor's" else 0,
        "Education Level_Master's": 1 if education == "Master's" else 0,
        "Education Level_PhD": 1 if education == "PhD" else 0,
        "Occupation_Employed": 1 if occupation == "Employed" else 0,
        "Occupation_Self-Employed": 1 if occupation == "Self-Employed" else 0,
        "Occupation_Unemployed": 1 if occupation == "Unemployed" else 0,
        "Location_Urban": 1 if location == "Urban" else 0,
        "Location_Suburban": 1 if location == "Suburban" else 0,
        "Location_Rural": 1 if location == "Rural" else 0,
        "Policy Type_Basic": 1 if policy_type == "Basic" else 0,
        "Policy Type_Comprehensive": 1 if policy_type == "Comprehensive" else 0,
        "Policy Type_Premium": 1 if policy_type == "Premium" else 0,
        "Customer Feedback_Poor": 1 if feedback == "Poor" else 0,
        "Customer Feedback_Average": 1 if feedback == "Average" else 0,
        "Customer Feedback_Good": 1 if feedback == "Good" else 0,
        "Smoking Status_Yes": 1 if smoking == "Yes" else 0,
        "Smoking Status_No": 1 if smoking == "No" else 0,
        "Exercise Frequency_Daily": 1 if exercise == "Daily" else 0,
        "Exercise Frequency_Weekly": 1 if exercise == "Weekly" else 0,
        "Exercise Frequency_Monthly": 1 if exercise == "Monthly" else 0,
        "Exercise Frequency_Rarely": 1 if exercise == "Rarely" else 0,
        "Property Type_House": 1 if property_type == "House" else 0,
        "Property Type_Condo": 1 if property_type == "Condo" else 0,
        "Property Type_Apartment": 1 if property_type == "Apartment" else 0,
    }

    # Merge into final input vector
    input_data.update(cats)

    input_df = pd.DataFrame([input_data])

    # Predict using MLflow model
    prediction = model.predict(input_df)[0]

    st.success(f"💡 Predicted Insurance Premium: **${prediction:,.2f}**")
