"""
Registers the best model in the MLflow Model Registry and promotes it to Production.
"""

import mlflow
from mlflow.tracking import MlflowClient
import json

def main():
    with open("data/best_model_uri.json") as f:
        data = json.load(f)
    best_model_uri = data["best_model_uri"]
    model_name = "Global_Salary_Prediction_Model"
    client = MlflowClient()
    print(f"Registering model: {model_name}")
    result = mlflow.register_model(model_uri=best_model_uri, name=model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Model '{model_name}' version {result.version} promoted to Production.")

if __name__ == "__main__":
    main()
