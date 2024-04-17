from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy import DummyOperator
from stage_01_data_ingestion import DataIngestionTrainingPipeline

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "data_ingestion_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag_ingestion:

    start = DummyOperator(task_id="start_ingestion")

    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=DataIngestionTrainingPipeline().main,
    )

    trigger_validation_dag = TriggerDagRunOperator(
        task_id="trigger_data_validation_dag",
        trigger_dag_id="data_validation_dag",
    )

    start >> data_ingestion >> trigger_validation_dag
