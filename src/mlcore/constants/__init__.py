from pathlib import Path

"""Contains path to configuration files"""

CONFIG_FILE_PATH = Path("/opt/airflow/dags/mlcore/config/config.yaml")
PARAMS_FILE_PATH = Path("/opt/airflow/dags/mlcore/constants/params.yaml")
SCHEMA_FILE_PATH = Path("/opt/airflow/dags/mlcore/constants/schema.yaml")
GCSKEY_FILE_PATH = Path("/opt/airflow/config/gcs_key.json")
