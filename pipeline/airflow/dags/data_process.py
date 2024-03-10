from datetime import datetime, timedelta, date
import pytz

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 2, tzinfo=pytz.timezone('US/Eastern')),
    # 'email': ['airflow@example.com'],
    # 'email_on_failure': False,
    # 'email_on_retry': False,
    # 'retries': 1,
    # 'retry_delay': timedelta(minutes=5),
}

# A DAG represents a workflow, a collection of tasks
with DAG(dag_id="Akhil_Test_Dag", default_args=default_args, catchup=False, schedule_interval='*/1 * * * *',) as dag:
    # Tasks are represented as operators
    hello = BashOperator(task_id="hello", bash_command="echo hello")

    @task()
    def airflow():
        print("airflow")

    # Set dependencies between tasks
    hello >> airflow()