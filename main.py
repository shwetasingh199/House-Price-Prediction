import joblib
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import process_features
from src.train_model import train_models
from src.evaluate import evaluate_model

# Step 1: Load dataset
df = load_data("data/raw/housing.csv")

# Step 2: Clean
df = clean_data(df)

# Step 3: Features
X, y, preprocessor = process_features(df)

# Step 4: Train
models, X_test, y_test = train_models(X, y, preprocessor)

# Step 5: Evaluate
for name, model in models.items():
    mae, rmse, r2 = evaluate_model(model, X_test, y_test)

    print(f"\n{name} Performance:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

# Step 6: Save best model
best_model = models["RandomForest"]
joblib.dump(best_model, "models/house_price_model.pkl")

print("\nModel saved successfully!")s