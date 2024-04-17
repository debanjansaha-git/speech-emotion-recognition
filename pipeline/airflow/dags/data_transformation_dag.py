from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from stage_03_data_transformation import DataTransformationTrainingPipeline

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "data_transformation_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag_transformation:

    training_data_transformation = PythonOperator(
        task_id="training_data_transformation",
        python_callable=DataTransformationTrainingPipeline(stage='train').main,
    )

    test_data_transformation = PythonOperator(
        task_id="test_data_transformation",
        python_callable=DataTransformationTrainingPipeline(stage='test').main,
    )

    trigger_training_dag = TriggerDagRunOperator(
        task_id="trigger_model_training_dag",
        trigger_dag_id="model_training_dag",
    )

    # Here, you might want to trigger another DAG or end the pipeline.
    # If ending, use a DummyOperator or similar as a visual endpoint.
    training_data_transformation >> test_data_transformation >> trigger_training_dag
