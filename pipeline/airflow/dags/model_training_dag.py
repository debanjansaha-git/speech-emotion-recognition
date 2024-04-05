from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from stage_04_model_trainer import ModelTrainerTrainingPipeline

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "model_training_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag_training:

    model_training = PythonOperator(
        task_id="model_training",
        python_callable=ModelTrainerTrainingPipeline(hypertune=False, epochs=2).main,
    )

    hyper_parameter_tuning = PythonOperator(
        task_id="hyper_parameter_tuning",
        python_callable=ModelTrainerTrainingPipeline(hypertune=True, epochs=10).main,
    )


    # Here, you might want to trigger another DAG or end the pipeline.
    # If ending, use a DummyOperator or similar as a visual endpoint.
    model_training >> hyper_parameter_tuning
