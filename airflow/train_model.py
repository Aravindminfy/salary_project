import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import joblib
import tempfile

# === Configuration ===
DATA_PATH = "/opt/airflow/dags/data/fe_salary_data.csv"
OUTPUT_URI_FILE = "/opt/airflow/dags/data/best_model_uri.json"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "Global_Salary_Prediction"

# === Step 1: Data Splitting ===
def train_test_split_data(df, target_col='total_salary', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# === Step 2: Model Definitions ===
def get_models_and_params():
    return {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [100], "max_depth": [10]}
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [100], "learning_rate": [0.1]}
        ),
        "XGBoost": (
            xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
            {"n_estimators": [100], "learning_rate": [0.1]}
        ),
    }

# === Step 3: Training and Logging ===
def run_pipeline_with_mlflow(X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_rmse = float("inf")
    best_model_uri = ""
    best_run_id = ""

    for name, (model, param_grid) in get_models_and_params().items():
        print(f"🔍 Training model: {name}")
        with mlflow.start_run(run_name=name) as run:
            if param_grid:
                grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
                mlflow.log_params(best_params)
            else:
                best_model = model.fit(X_train, y_train)
                best_params = {}

            preds = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

            # ✅ Save model manually and log it as artifact
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = os.path.join(tmp_dir, "model.joblib")
                joblib.dump(best_model, model_path)
                mlflow.log_artifact(model_path, artifact_path="model")

            print(f"✅ {name} -> RMSE: {rmse:.2f}, R²: {r2:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_uri = f"runs:/{run.info.run_id}/model"
                best_run_id = run.info.run_id

    return best_model_uri, best_run_id, best_rmse

# === Step 4: Main Execution ===
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if 'total_salary' not in df.columns:
        raise ValueError("❌ 'total_salary' column not found in dataset.")

    X_train, X_test, y_train, y_test = train_test_split_data(df)

    best_model_uri, best_run_id, best_rmse = run_pipeline_with_mlflow(X_train, X_test, y_train, y_test)

    os.makedirs(os.path.dirname(OUTPUT_URI_FILE), exist_ok=True)
    with open(OUTPUT_URI_FILE, "w") as f:
        json.dump({"best_model_uri": best_model_uri, "best_run_id": best_run_id}, f)

    print("\n🏆 Best Model Info:")
    print(f"Best Run ID: {best_run_id}")
    print(f"Best Model URI: {best_model_uri}")
    print(f"Best RMSE: {best_rmse:.2f}")
    print(f"📁 URI saved to: {OUTPUT_URI_FILE}")

if __name__ == "__main__":
    main()
