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

## 📂 Project Structure  

```
insurance-premium-prediction/
│── data/
│   ├── train.csv
│   ├── test.csv
│── notebooks/
│   ├── eda.ipynb
│── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│── app.py                # Streamlit App
│── requirements.txt
│── Dockerfile
│── README.md
```

---

## ⚙️ Setup Instructions  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/insurance-premium-prediction.git
cd insurance-premium-prediction
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
python src/train.py
```

---

## 🌐 Run Streamlit App  

Run the app locally:  
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.  

---

## 🐳 Run with Docker  

### 1️⃣ Build Docker Image  
```bash
docker build -t insurance-premium-app .
```

### 2️⃣ Run Container  
```bash
docker run -p 8501:8501 insurance-premium-app
```

Now access the app at [http://localhost:8501](http://localhost:8501).  

---

## 📊 Example Prediction  

Input:  
- Age: 30  
- Income: 50,000  
- Marital Status: Married  
- Policy: Comprehensive  

Output:  
```
💡 Predicted Premium: $3,420.75
```

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **Scikit-learn, XGBoost** (ML models)  
- **MLflow** (experiment tracking & model registry)  
- **Streamlit** (frontend)  
- **Docker** (containerization)  

---

## 🔮 Next Steps  
- Add SHAP explainability for premium predictions.  
- Integrate with a cloud MLflow server (AWS/GCP/Azure).  
- Deploy Streamlit app to **Streamlit Cloud** or **Heroku**.  

---

🙌 **Contributions Welcome!** Fork the repo, open issues, and submit PRs.  
