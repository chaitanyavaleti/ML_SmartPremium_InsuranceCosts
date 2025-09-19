import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("playground-series-s4e12/train.csv")
train.drop(['id', 'Policy Start Date'], axis=1, inplace=True)

target = "Premium Amount"

Q1 = train["Premium Amount"].quantile(0.25)
Q3 = train["Premium Amount"].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train = train[(train["Premium Amount"] >= lower_bound) & (train["Premium Amount"] <= upper_bound)]

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

# Categorical pipeline: impute → one-hot encode
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

X = train.drop(columns=[target])
y = train[target]

# Identify column types
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(exclude=["int64","float64"]).columns

# Combine numeric + categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_estimators=100,       # start baseline
    learning_rate=0.05,     # baseline learning rate
    max_depth=5,            # baseline depth
    subsample=0.8,
    colsample_bytree=0.8
))
])

param_grid = {
    "model__n_estimators": [100, 200, 500],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [4, 5, 6, 8],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0]
}

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="r2",   # optimize for R²
    cv=3,           # 3-fold cross-validation
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best R²:", grid_search.best_score_)

best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_val)

print("Validation R²:", r2_score(y_val, y_pred))
print("Validation RMSE:", mean_squared_error(y_val, y_pred, squared=False))
print("Validation MAE:", mean_absolute_error(y_val, y_pred))

