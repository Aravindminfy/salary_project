"""
Loads the production model and makes predictions on new data.
"""

import mlflow
import mlflow.sklearn
import pandas as pd

def main():
    model_name = "Global_Salary_Prediction_Model"
    model_uri = f"models:/{model_name}@Production"
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    # For demonstration, use the first 5 rows of the latest feature-engineered data
    df = pd.read_csv("data/fe_salary_data.csv")
    X = df.drop(columns=['total_salary'])
    preds = model.predict(X.head(5))
    for i, pred in enumerate(preds, 1):
        print(f"Sample {i}: Predicted Salary = {pred:,.2f}")

if __name__ == "__main__":
    main()
