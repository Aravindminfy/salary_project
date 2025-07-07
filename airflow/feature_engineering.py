"""
Feature engineering: encoding, imputation, and scaling.
"""

import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def knn_impute_categoricals(df, cat_cols, n_neighbors=5):
    df_copy = df.copy()
    required_cols = list(set(cat_cols + [col for col in df.columns if df[col].dtype != 'object' or col in cat_cols]))
    df_numeric = df_copy[required_cols].copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_numeric[col] = df_numeric[col].astype(str)
        df_numeric[col] = le.fit_transform(df_numeric[col])
        encoders[col] = le
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df_numeric)
    df_imputed_numeric = pd.DataFrame(imputed_array, columns=df_numeric.columns)
    for col in cat_cols:
        df_imputed_numeric[col] = df_imputed_numeric[col].round().astype(int)
        df_imputed_numeric[col] = encoders[col].inverse_transform(df_imputed_numeric[col])
    for col in cat_cols:
        df_copy[col] = df_imputed_numeric[col]
    return df_copy

def preprocess_categorical(df, target_col='total_salary'):
    experience_mapping = {'Entry': 0, 'Mid': 1, 'Senior': 2, 'Lead': 3}
    company_size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    if 'experience_level' in df.columns:
        df['experience_level'] = df['experience_level'].fillna('Mid').map(experience_mapping)
    if 'company_size' in df.columns:
        df['company_size'] = df['company_size'].fillna('Medium').map(company_size_mapping)
    for col in ['employment_type', 'salary_currency', 'currency']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    ohe_cols = [col for col in ['employment_type', 'salary_currency', 'currency'] if col in df.columns]
    if ohe_cols:
        ohe = OneHotEncoder(sparse_output=False, drop='if_binary')
        encoded = ohe.fit_transform(df[ohe_cols])
        ohe_names = ohe.get_feature_names_out(ohe_cols)
        df_ohe = pd.DataFrame(encoded, columns=ohe_names, index=df.index)
        df = pd.concat([df.drop(ohe_cols, axis=1), df_ohe], axis=1)
    if 'job_title' in df.columns:
        means = df.groupby('job_title')[target_col].mean()
        df['job_title_target_enc'] = df['job_title'].map(means)
        df = df.drop('job_title', axis=1)
    if 'company_location' in df.columns:
        freq = df['company_location'].value_counts(normalize=True)
        df['company_location_freq_enc'] = df['company_location'].map(freq)
        df = df.drop('company_location', axis=1)
    return df

def build_preprocessing(df, target_col='total_salary'):
    # Impute categoricals
    cat_cols = [col for col in ['experience_level', 'employment_type'] if col in df.columns]
    df = knn_impute_categoricals(df, cat_cols)
    df = preprocess_categorical(df, target_col)
    # Numeric columns (excluding target)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols)
    ], remainder='passthrough')
    transformed_array = preprocessor.fit_transform(df)
    passthrough_cols = [col for col in df.columns if col not in numeric_cols]
    all_cols = numeric_cols + passthrough_cols
    df_transformed = pd.DataFrame(transformed_array, columns=all_cols, index=df.index)
    return df_transformed

def main():
    input_path = '/opt/airflow/dags/data/cleaned_salary_data.csv'
    output_dir = '/opt/airflow/dags/data'
    output_path = os.path.join(output_dir, 'fe_salary_data.csv')
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df = build_preprocessing(df, target_col='total_salary')
    df.to_csv(output_path, index=False)
    print(f"✅ Feature engineering complete. Saved to {output_path}")

if __name__ == "__main__":
    main()
