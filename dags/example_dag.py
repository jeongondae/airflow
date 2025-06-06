from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


with DAG(dag_id="example_dag", start_date=datetime(2025,6,1), schedule_interval="@daily", catchup=False) as dag:
    t1 = BashOperator(task_id="print_date", bash_command="date")
    t1
