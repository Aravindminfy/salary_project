import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def drop_useless_columns(df):
    for col in ['education', 'skills']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df

def clean_job_titles(df):
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
    if 'job_title' in df.columns:
        df['job_title'] = df['job_title'].replace(title_map)
    return df

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
        df_copy[col] = df_imputed_numeric[col]
    return df_copy

def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])
    return df

def preprocess_categorical(df, target_col='total_salary'):
    df = df.copy()
    experience_mapping = {'Entry': 0, 'Mid': 1, 'Senior': 2, 'Lead': 3}
    df['experience_level'] = df['experience_level'].map(experience_mapping).fillna(1)
    company_size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['company_size'] = df['company_size'].map(company_size_mapping).fillna(1)
    one_hot_cols = ['employment_type', 'salary_currency', 'currency']
    for col in one_hot_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    ohe = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')
    encoded = ohe.fit_transform(df[one_hot_cols])
    ohe_cols = ohe.get_feature_names_out(one_hot_cols)
    df_ohe = pd.DataFrame(encoded, columns=ohe_cols, index=df.index)
    df = pd.concat([df.drop(one_hot_cols, axis=1), df_ohe], axis=1)
    if 'job_title' in df.columns and target_col in df.columns:
        means = df.groupby('job_title')[target_col].mean()
        df['job_title_target_enc'] = df['job_title'].map(means)
        df = df.drop('job_title', axis=1)
    if 'company_location' in df.columns:
        freq = df['company_location'].value_counts(normalize=True)
        df['company_location_freq_enc'] = df['company_location'].map(freq)
        df = df.drop('company_location', axis=1)
    return df

def build_preprocessing(df, target_col='total_salary'):
    df = drop_useless_columns(df)
    df = clean_job_titles(df)
    df = knn_impute_categoricals(df, ['experience_level', 'employment_type'])
    for col in ['base_salary', 'adjusted_total_usd', 'salary_in_usd']:
        if col in df.columns:
            df = cap_outliers_iqr(df, col)
    if 'base_salary' in df.columns:
        df['log_base_salary'] = np.log1p(df['base_salary'])
    df = preprocess_categorical(df, target_col)
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

def preprocess_user_input(user_json, target_col='total_salary'):
    user_df = pd.DataFrame([user_json])
    user_df[target_col] = 0  # Dummy target for encoding
    processed = build_preprocessing(user_df, target_col)
    if target_col in processed.columns:
        processed = processed.drop(target_col, axis=1)
    return processed
