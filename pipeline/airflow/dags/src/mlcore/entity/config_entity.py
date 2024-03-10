from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_ravdess_dir: Path
    unzip_tess_dir: Path
    unzip_cremad_dir: Path
    unzip_savee_dir: Path
    local_output_path: Path
    validation_status: Path
    metadata_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    metadata_path: Path
    output_path: Path
    train_path: Path
    test_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    model_params: dict
    target_col: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_path: Path
    model_params: dict
    metric_file_name: str
    target_col: str
    mlflow_uri: str
