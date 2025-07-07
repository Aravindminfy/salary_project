FROM apache/airflow:2.7.1-python3.9

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --upgrade pip setuptools wheel

# Install all your needed Python packages here
RUN pip install \
    pandas \
    numpy \
    scikit-learn \
    psycopg2-binary \
    pyyaml \
    mlflow \
    xgboost \
    joblib \
    evidently

RUN pip install flask-session apache-airflow-providers-postgres apache-airflow-providers-google apache-airflow-providers-amazon

RUN pip install pandas numpy scikit-learn psycopg2-binary pyyaml mlflow xgboost joblib evidently
COPY ./dags /opt/airflow/dags
COPY ./plugins /opt/airflow/plugins
