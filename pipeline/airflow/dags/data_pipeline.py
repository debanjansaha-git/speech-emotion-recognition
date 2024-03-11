from datetime import datetime, timedelta, date
import pytz

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python_operator import PythonOperator
from airflow import configuration as conf

from stage_01_data_ingestion import DataIngestionTrainingPipeline
from stage_02_data_validation import DataValidationTrainingPipeline
# from stage_03_data_transformation import DataTransformationTrainingPipeline
# from src.data_collection.data_collection import (
#     prepare_folders,
#     copy_ravdess_dataset,
#     copy_meld_dataset,
# )

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set("core", "enable_xcom_pickling", "True")


# Define Python Functions
def python_function():
    obj = DataIngestionTrainingPipeline()
    obj.main()


def copy_data():
    print("Blank function")
    # prepare_folders()
    # copy_ravdess_dataset()


# Start Dag Definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime.now(),
    # 'email': ['airflow@example.com'],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# A DAG represents a workflow, a collection of tasks
dag = DAG(
    "Data_Pipeline",
    description="This DAG represents the Data Pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

# Define Operators

start_pipeline = BashOperator(task_id="start_pipeline", bash_command="pwd && ls -lart")

download_data = PythonOperator(
    task_id="download_data",
    python_callable=DataIngestionTrainingPipeline().main,
    provide_context=True,
    dag=dag,
)

data_validation = PythonOperator(
    task_id="data_validation",
    python_callable=DataValidationTrainingPipeline().main,
    provide_context=True,
    dag=dag,
)

authenticate = PythonOperator(
    task_id="authenticate", 
    python_callable=authenticate, 
    op_args= ["speech-emotion-recognition/pipeline/airflow/secrets/gcs_key.json"],
    dag=dag
)

upload_directory = PythonOperator(
    task_id="upload_directory", 
    python_callable=upload_directory,
    op_args = ['mlops-grp3-data-bucket', 'mlops-group3/data'] ,
    dag=dag
)


# data_transformation = PythonOperator(
#     task_id="data_transformation",
#     python_callable=DataTransformationTrainingPipeline().main,
#     provide_context=True,
#     dag=dag,
# )

# upload_data = PythonOperator(
#     task_id="upload_data",
#     python_callable=python_function,
#     provide_context=True,
#     dag=dag,
# )

# end_pipeline = EmptyOperator(
#     task_id="end_pipeline",
#     dag=dag,
# )

# Set dependencies between tasks
(
    start_pipeline
    >> download_data
    >> data_validation
    # >> data_transformation
    # >> upload_data
    # >> end_pipeline
)
