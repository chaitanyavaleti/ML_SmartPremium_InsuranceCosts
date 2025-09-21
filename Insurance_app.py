import streamlit as st
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

MODEL_NAME = "InsurancePremiumPrediction"
client = MlflowClient()

# Find the latest version tagged as Production
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
prod_versions = [v for v in versions if v.tags.get("stage") == "Production"]

if not prod_versions:import streamlit as st
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

MODEL_NAME = "InsurancePremiumPrediction"
client = MlflowClient()

# Find the latest version tagged as Production
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
prod_versions = [v for v in versions if v.tags.get("stage") == "Production"]

if not prod_versions:
    raise RuntimeError("No model version tagged as Production!")

# Pick latest Production version
latest_prod_version = max(prod_versions, key=lambda v: int(v.version))

model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{latest_prod_version.version}"
)

st.set_page_config(page_title="Insurance Premium Predictor", layout="wide")

st.title("💰 Insurance Premium Prediction App")

st.header("📂 Bulk Prediction from CSV")

uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    try:
        predictions = model.predict(df)
        df["Predicted Premium"] = predictions
        st.subheader("✅ Predictions")
        st.write(df.head())

        # Download button
        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            data=csv_download,
            file_name="predicted_premiums.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error while predicting: {e}")

    raise RuntimeError("No model version tagged as Production!")

# Pick latest Production version
latest_prod_version = max(prod_versions, key=lambda v: int(v.version))

model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{latest_prod_version.version}"
)

st.set_page_config(page_title="Insurance Premium Predictor", layout="wide")

st.title("💰 Insurance Premium Prediction App")

# -------------------------------
# Input Form
# -------------------------------
with st.form("premium_form"):

    col1, col2 = st.columns(2)

    # Numeric inputs
    with col1: 
        st.subheader("Customer Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income", min_value=1000, max_value=1000000, value=50000)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
        health_score = st.slider("Health Score", 0, 100, 75)
        prev_claims = st.number_input("Previous Claims", min_value=0, max_value=20, value=0)
        vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        ins_duration = st.number_input("Insurance Duration (years)", min_value=0, max_value=50, value=5)

    with col2:
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

    input_data = {
    "Age": float(age),
    "Annual Income": float(income),
    "Number of Dependents": float(dependents),
    "Health Score": float(health_score),
    "Previous Claims": float(prev_claims),
    "Vehicle Age": float(vehicle_age),
    "Credit Score": float(credit_score),
    "Insurance Duration": float(ins_duration),
    "Gender": gender,  # string
    "Marital Status": marital_status,  # string
    "Education Level": education,  # string
    "Occupation": occupation,  # string
    "Location": location,  # string
    "Policy Type": policy_type,  # string
    "Customer Feedback": feedback,  # string
    "Smoking Status": smoking,  # string
    "Exercise Frequency": exercise,  # string
    "Property Type": property_type  # string
    }

    input_df = pd.DataFrame([input_data])

    training_columns = pd.read_csv("training_columns.csv", header=None)[0].tolist()

    numeric_cols = [
        "Age", "Annual Income", "Number of Dependents", "Health Score",
        "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration"
    ]
    input_df[numeric_cols] = input_df[numeric_cols].astype(float)

    categorical_cols = [
        "Gender", "Marital Status", "Education Level", "Occupation", "Location",
        "Policy Type", "Customer Feedback", "Smoking Status", "Exercise Frequency", "Property Type"
    ]

    input_df[categorical_cols] = input_df[categorical_cols].astype(str)

    # Predict using MLflow model
    prediction = model.predict(input_df)[0]
    prediction = float(prediction)

    st.success(f"💡 Predicted Insurance Premium: **${prediction:,.2f}**")
