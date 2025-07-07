import mlflow
import mlflow.sklearn
import pandas as pd

def load_production_model(model_name="Global_Salary_Prediction_Model"):
    mlflow.set_tracking_uri("http://localhost:5000")
    model_uri = f"models:/{model_name}@production"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def align_features(processed, model):
    """
    Aligns the processed DataFrame columns with the model's expected features.
    Adds missing columns as 0, removes extra columns, and orders correctly.
    """
    # For sklearn >=1.0
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        # Fallback: try to get from booster (xgboost/lightgbm) or infer from training
        expected_features = list(processed.columns)  # fallback, may need manual fix

    # Add missing columns as 0
    for col in expected_features:
        if col not in processed.columns:
            processed[col] = 0

    # Remove extra columns
    processed = processed[expected_features]
    return processed
