import os
from pathlib import Path

"""Contains path to configuration files"""

if os.getenv["RUNTIME"] == "local":
    CONFIG_FILE_PATH = Path("src/mlcore/config/config_local.yaml")
    PARAMS_FILE_PATH = Path("src/mlcore/constants/params.yaml")
    SCHEMA_FILE_PATH = Path("src/mlcore/constants/schema.yaml")
    GCSKEY_FILE_PATH = Path("/opt/airflow/config/gcs_key.json")
else:
    CONFIG_FILE_PATH = Path("mlcore/config/config.yaml")
    PARAMS_FILE_PATH = Path("mlcore/constants/params.yaml")
    SCHEMA_FILE_PATH = Path("mlcore/constants/schema.yaml")
    # GCSKEY_FILE_PATH = Path("/opt/airflow/config/gcs_key.json")
