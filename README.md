# 📘 Insurance Premium Prediction  

This project builds a **Machine Learning (ML)** model to predict **insurance premiums** based on customer demographics, financial, health, and policy-related details. It integrates with **MLflow** for experiment tracking and model management, and provides a **Streamlit app** for real-time predictions.  

---

## 🚀 Features  
- **Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling.  
- **Regression Models**: Trains Linear Regression, Decision Tree, Random Forest, and XGBoost.  
- **Model Tracking**: MLflow used for training, evaluation, and automatic model registration.  
- **Best Model Deployment**: Best model automatically pushed to MLflow Model Registry.  
- **Streamlit Frontend**: User-friendly interface to predict insurance premium amounts.  
- **Dockerized Deployment**: Ready to run anywhere with Docker.  

---

## ⚙️ Setup Instructions  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/chaitanyavaleti/ML_SmartPremium_InsuranceCosts.git
cd ML_SmartPremium_InsuranceCosts
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run MLflow Tracking Server (Optional)  
```bash
mlflow ui
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 🏋️ Train Models with MLflow  

Run training script to:  
- Preprocess data  
- Train regression models  
- Log results with MLflow  
- Register best model automatically  

```bash
python SmartInsurance_MLFlow.py
```

---

## 🌐 Run Streamlit App  

Run the app locally:  
```bash
streamlit run Insurance_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.  


## 📊 Example Prediction  

<img width="1870" height="772" alt="image" src="https://github.com/user-attachments/assets/282fba0a-e5c1-433f-82f3-0429cbc2f570" />


---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **Scikit-learn, XGBoost** (ML models)  
- **MLflow** (experiment tracking & model registry)  
- **Streamlit** (frontend)  
- **Docker** (containerization)  

## 👤 Created By

Chaitanya Valeti (MAE4 [AIML-C-WD-E-B18)
Built as a Mini project for the **AIML (Artificial Intelligence & Machine Learning)** domain at GUVI (HCL Tech).
