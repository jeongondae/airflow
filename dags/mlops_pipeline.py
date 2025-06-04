from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG("mlops_dvc_pipeline",
         schedule_interval=None,
         start_date=datetime(2024, 1, 1),
         catchup=False) as dag:

    download = BashOperator(
        task_id='download_data',
        bash_command='dvc repro download',
        cwd='/opt/airflow'  # airflow-docker 내 경로로 맞춰야 함
    )

    preprocess = BashOperator(
        task_id='preprocess_data',
        bash_command='dvc repro preprocess',
        cwd='/opt/airflow'
    )

    train = BashOperator(
        task_id='train_model',
        bash_command='dvc repro train',
        cwd='/opt/airflow'
    )

    evaluate = BashOperator(
        task_id='evaluate_model',
        bash_command='dvc repro evaluate',
        cwd='/opt/airflow'
    )

    download >> preprocess >> train >> evaluate
