"""
Performs EDA and preprocessing: missing value handling, outlier capping, and basic cleaning.
"""

import pandas as pd
import numpy as np
import os

def drop_useless_columns(df):
    # Drop columns with 100% missing values or irrelevant columns
    drop_cols = [col for col in ['education', 'skills'] if col in df.columns]
    return df.drop(columns=drop_cols, errors='ignore')

def clean_job_titles(df):
    # Standardize job titles
    title_map = {
        'Sofware Engneer': 'Software Engineer',
        'Softwre Engineer': 'Software Engineer',
        'Software Engr': 'Software Engineer',
        'ML Engr': 'ML Engineer',
        'ML Enginer': 'ML Engineer',
        'Dt Scientist': 'Data Scientist',
        'Data Scntist': 'Data Scientist',
        'Data Scienist': 'Data Scientist'
    }
    df['job_title'] = df['job_title'].replace(title_map)
    return df

def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])
    return df

def main():
    input_path = '/opt/airflow/dags/data/raw_salary_data.csv'
    output_dir = '/opt/airflow/dags/data'
    output_path = os.path.join(output_dir, 'cleaned_salary_data.csv')
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df = drop_useless_columns(df)
    df = clean_job_titles(df)
    for col in ['base_salary', 'adjusted_total_usd', 'salary_in_usd']:
        if col in df.columns:
            df = cap_outliers_iqr(df, col)
    if 'base_salary' in df.columns:
        df['log_base_salary'] = np.log1p(df['base_salary'])
    df.to_csv(output_path, index=False)
    print(f"✅ EDA and preprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    main()
