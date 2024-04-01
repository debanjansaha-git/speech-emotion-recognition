from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from stage_02_data_validation import DataValidationTrainingPipeline

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "data_validation_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag_validation:

    data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=DataValidationTrainingPipeline().main,
    )

    trigger_transformation_dag = TriggerDagRunOperator(
        task_id="trigger_data_transformation_dag",
        trigger_dag_id="data_transformation_dag",
    )

    data_validation >> trigger_transformation_dag
