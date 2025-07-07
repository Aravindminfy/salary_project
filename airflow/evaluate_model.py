"""
Evaluates the best model on the test set and logs metrics to MLflow and registers the model.
"""

import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Paths
    data_dir = "/opt/airflow/dags/data"
    fe_data_path = os.path.join(data_dir, "fe_salary_data.csv")
    split_idx_path = os.path.join(data_dir, "split_indices.json")
    best_model_uri_path = os.path.join(data_dir, "best_model_uri.json")

    # Load data
    df = pd.read_csv(fe_data_path)
    X = df.drop(columns=['total_salary'])
    y = df['total_salary']

    # Split test data
    if os.path.exists(split_idx_path):
        with open(split_idx_path) as f:
            split_indices = json.load(f)
        test_idx = split_indices["test_idx"]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
    else:
        X_train = X.sample(frac=0.8, random_state=42)
        X_test = X.drop(X_train.index)
        y_test = y.loc[X_test.index]

    # Set MLflow tracking
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Global_Salary_Prediction")

    # Load best model URI
    try:
        with open(best_model_uri_path) as f:
            model_info = json.load(f)
            best_model_uri = model_info["best_model_uri"]
            best_run_id = model_info.get("best_run_id")
    except Exception as e:
        raise Exception(f"❌ Failed to load best model URI: {e}")

    # Try loading the model
    try:
        model = mlflow.sklearn.load_model(best_model_uri)
    except MlflowException as e:
        raise Exception(f"❌ MLflow could not load model from URI: {best_model_uri}\nError: {str(e)}")

    # Predict & evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"✅ Evaluation -> RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Start new run and log metrics
    with mlflow.start_run(run_name="Evaluation"):
        mlflow.log_metric("eval_rmse", rmse)
        mlflow.log_metric("eval_r2", r2)

        # Register model
        model_name = "GlobalSalaryModel"
        print(f"🔄 Registering model as '{model_name}'")
        try:
            model_details = mlflow.register_model(model_uri=best_model_uri, name=model_name)
            print(f"✅ Registered model to MLflow Model Registry: {model_details.name} v{model_details.version}")
        except Exception as e:
            print(f"⚠️ Failed to register model: {e}")

if __name__ == "__main__":
    main()
