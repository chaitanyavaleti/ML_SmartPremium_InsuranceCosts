import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================
# 1. Load Data
# ==========================
train = pd.read_csv("playground-series-s4e12/train.csv")
test = pd.read_csv("playground-series-s4e12/test.csv")

train.drop(['id', 'Policy Start Date'], axis=1, inplace=True)
test.drop(['id', 'Policy Start Date'], axis=1, inplace=True)

target = "Premium Amount"

X = train.drop(columns=[target])
y = train[target]

# Identify column types
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(exclude=["int64","float64"]).columns

# ==========================
# 2. Preprocessing Pipeline
# ==========================
# Numeric pipeline: impute → scale
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

# Categorical pipeline: impute → one-hot encode
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine numeric + categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# ==========================
# 3. Train/Val Split
# ==========================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# 4. Models Dictionary
# ==========================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# ==========================
# 5. MLflow Training & Logging
# ==========================
mlflow.set_experiment("Insurance_Premium_Prediction_Pipeline")

best_rmse = float("inf")
best_model = None
best_model_name = None
best_run_id = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        
        # Full pipeline = preprocessing + model
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # Train
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        # Log parameters & metrics
        mlflow.log_param("model", name)
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Save the pipeline (not just the model)
        mlflow.sklearn.log_model(pipe, artifact_path="pipeline_model")

        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name
            best_run_id = run.info.run_id

model_uri = f"runs:/{best_run_id}/model"

mlflow.register_model(model_uri=model_uri, name="InsurancePremiumModel")

print(f"✅ Best model registered: {best_model_name} with RMSE={best_rmse:.2f} and Run ID={best_run_id}")