from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from stage_00_data_generation import DataGenerationTrainingPipeline

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "data_generation_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag_generation:

    data_generation_0 = PythonOperator(
        task_id="data_generation_zero",
        python_callable=DataGenerationTrainingPipeline().main,
    )

    data_generation_1 = PythonOperator(
        task_id="data_generation_one",
        python_callable=DataGenerationTrainingPipeline().main,
    )

    trigger_transformation_dag = TriggerDagRunOperator(
        task_id="trigger_data_transformation_dag",
        trigger_dag_id="data_transformation_dag",
    )

    # Here, you might want to trigger another DAG or end the pipeline.
    # If ending, use a DummyOperator or similar as a visual endpoint.
    data_generation_0 >> data_generation_1 >> trigger_transformation_dag
