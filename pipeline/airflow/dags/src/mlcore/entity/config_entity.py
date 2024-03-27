from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataGenerationConfig:
    """
    Data class for data generation configuration.

    Summary:
        This data class represents the configuration for data generation.

    Explanation:
        The DataIngestionConfig class is a frozen data class that holds the configuration parameters for data ingestion.
        It contains the root directory path, source URL, and local data path for data ingestion.

    Attributes:
        root_dir (Path): The root directory path for data ingestion.
        source_URL (str): The URL of the data source.
        local_data_path (Path): The local path where the data will be saved after ingestion.

    Examples:
        config = DataIngestionConfig(root_dir, source_URL, local_data_path)
    """

    gcp_metadata_bucket: str
    gcp_train_bucket: str
    gcp_test_bucket: str
    metadata_dir: Path
    train_dir: Path
    test_dir: Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Data class for data ingestion configuration.

    Summary:
        This data class represents the configuration for data ingestion.

    Explanation:
        The DataIngestionConfig class is a frozen data class that holds the configuration parameters for data ingestion.
        It contains the root directory path, source URL, and local data path for data ingestion.

    Attributes:
        root_dir (Path): The root directory path for data ingestion.
        source_URL (str): The URL of the data source.
        local_data_path (Path): The local path where the data will be saved after ingestion.

    Examples:
        config = DataIngestionConfig(root_dir, source_URL, local_data_path)
    """

    root_dir: Path
    source_URL: str
    local_data_path: Path
    gcp_bucket_name: str
    gcp_data_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Data class for data validation configuration.

    Summary:
        This data class represents the configuration for data validation.

    Explanation:
        The DataValidationConfig class is a frozen data class that holds the configuration parameters for data validation.
        It contains the root directory path, directories for unzipping different datasets, local output path, validation status path,
        and the metadata schema for validation.

    Attributes:
        root_dir (Path): The root directory path for data validation.
        unzip_ravdess_dir (Path): The directory path for unzipping the RAVDESS dataset.
        unzip_tess_dir (Path): The directory path for unzipping the TESS dataset.
        unzip_cremad_dir (Path): The directory path for unzipping the CREMA-D dataset.
        unzip_savee_dir (Path): The directory path for unzipping the SAVEE dataset.
        local_output_path (Path): The local output path for validation results.
        validation_status (Path): The path for the validation status file.
        metadata_schema (dict): The metadata schema for validation.

    Examples:
        config = DataValidationConfig(root_dir, unzip_ravdess_dir, unzip_tess_dir, unzip_cremad_dir,
                                    unzip_savee_dir, local_output_path, validation_status, metadata_schema)
    """

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
    """
    Data class for data transformation configuration.

    Summary:
        This data class represents the configuration for data transformation.

    Explanation:
        The DataTransformationConfig class is a frozen data class that holds the configuration parameters for data transformation.
        It contains the root directory path, metadata path, output path, train path, and test path for data transformation.

    Attributes:
        root_dir (Path): The root directory path for data transformation.
        metadata_path (Path): The path to the metadata file.
        output_path (Path): The output path for the transformed data.
        train_path (Path): The path for the train data.
        test_path (Path): The path for the test data.

    Examples:
        config = DataTransformationConfig(root_dir, metadata_path, output_path, train_path, test_path)
    """

    root_dir: Path
    metadata_path: Path
    output_path: Path
    train_path: Path
    val_path: Path
    test_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Data class for model trainer configuration.

    Summary:
        This data class represents the configuration for model training.

    Explanation:
        The ModelTrainerConfig class is a frozen data class that holds the configuration parameters for model training.
        It contains the root directory path, paths for train and test data, model name, model parameters, and target column name.

    Attributes:
        root_dir (Path): The root directory path for model training.
        train_data_path (Path): The path to the train data.
        test_data_path (Path): The path to the test data.
        model_name (str): The name of the model.
        model_params (dict): The parameters for the model.
        target_col (str): The name of the target column.

    Examples:
        config = ModelTrainerConfig(root_dir, train_data_path, test_data_path, model_name, model_params, target_col)
    """

    root_dir: Path
    train_path: Path
    val_path: Path
    model_name: str
    params: dict
    target_col: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Data class for model evaluation configuration.

    Summary:
        This data class represents the configuration for model evaluation.

    Explanation:
        The ModelEvaluationConfig class is a frozen data class that holds the configuration parameters for model evaluation.
        It contains the root directory path, paths for train and test data, model path, model parameters, metric file name, target column name, and MLflow URI.

    Attributes:
        root_dir (Path): The root directory path for model evaluation.
        train_data_path (Path): The path to the train data.
        test_data_path (Path): The path to the test data.
        model_path (Path): The path to the trained model.
        model_params (dict): The parameters for the model.
        metric_file_name (str): The file name for the metric file.
        target_col (str): The name of the target column.
        mlflow_uri (str): The URI for MLflow.

    Examples:
        config = ModelEvaluationConfig(root_dir, train_data_path, test_data_path, model_path, model_params,
                                    metric_file_name, target_col, mlflow_uri)
    """

    root_dir: Path
    train_path: Path
    val_path: Path
    test_path: Path
    model_path: Path
    model_params: dict
    metric_file_name: str
    target_col: str
    mlflow_uri: str
