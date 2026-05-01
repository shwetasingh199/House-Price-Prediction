import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

# -------------------------------
# 1. LOAD CLEAN DATA
# -------------------------------
df = pd.read_csv("data/processed/cleaned_data.csv")

print("\n📊 Cleaned dataset loaded!")

# -------------------------------
# 2. DEFINE FEATURES & TARGET
# -------------------------------
X = df.drop("price", axis=1)
y = df["price"]

# Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# -------------------------------
# 3. PREPROCESSING PIPELINE
# -------------------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

# -------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Train-Test Split Done!")

# -------------------------------
# 5. MODELS
# -------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

if xgb_available:
    models["XGBoost"] = XGBRegressor(random_state=42)

# -------------------------------
# 6. HYPERPARAMETER TUNING (RF)
# -------------------------------
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5]
}

print("\n🔍 Tuning Random Forest...")

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

print("✅ Best Random Forest Params:", rf_grid.best_params_)

# -------------------------------
# 7. TRAIN ALL MODELS
# -------------------------------
trained_models = {}

for name, model in models.items():
    if name == "RandomForest":
        trained_models[name] = best_rf
        continue

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print(f"\n🚀 Training {name}...")
    pipeline.fit(X_train, y_train)

    trained_models[name] = pipeline

# -------------------------------
# 8. EVALUATION FUNCTION
# -------------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return mae, rmse, r2

# -------------------------------
# 9. MODEL COMPARISON
# -------------------------------
results = []

print("\n📊 Model Performance:\n")

for name, model in trained_models.items():
    mae, rmse, r2 = evaluate(model, X_test, y_test)

    print(f"{name}:")
    print(f" MAE  : {mae:.2f}")
    print(f" RMSE : {rmse:.2f}")
    print(f" R2   : {r2:.4f}\n")

    results.append((name, r2, model))

# -------------------------------
# 10. SELECT BEST MODEL
# -------------------------------
best_model_name, best_score, best_model = sorted(results, key=lambda x: x[1], reverse=True)[0]

print(f"🏆 Best Model: {best_model_name} (R2: {best_score:.4f})")

# -------------------------------
# 11. SAVE MODEL
# -------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/house_price_model.pkl")

print("\n💾 Model saved at models/house_price_model.pkl")