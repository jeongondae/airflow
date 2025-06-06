FROM apache/airflow:2.9.1
USER root
RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir dvc mlflow scikit-learn pandas joblib
USER airflow
