from mlcore.constants import *
from mlcore.utils.common import read_yaml, create_directories
import os
from mlcore.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    """
    Class for managing configuration.

    Summary:
        This class handles the management of configuration files and provides methods to retrieve specific configuration objects.

    Explanation:
        The ConfigurationManager class reads and manages configuration files for data ingestion, validation, transformation, model training, and model evaluation.
        The class takes optional file paths for the configuration, parameters, and schema files.
        It provides methods to retrieve specific configuration objects for data ingestion, data validation, data transformation, model training, and model evaluation.

    Methods:
        get_data_ingestion_config() -> List[DataIngestionConfig]:
            Retrieves a list of DataIngestionConfig objects for each data ingestion configuration.

        get_data_validation_config() -> DataValidationConfig:
            Retrieves the DataValidationConfig object for data validation.

        get_data_transformation_config() -> DataTransformationConfig:
            Retrieves the DataTransformationConfig object for data transformation.

        get_model_trainer_config() -> ModelTrainerConfig:
            Retrieves the ModelTrainerConfig object for model training.

        get_model_evaluation_config() -> ModelEvaluationConfig:
            Retrieves the ModelEvaluationConfig object for model evaluation.

    Raises:
        None.

    Examples:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_validation_config = config_manager.get_data_validation_config()
        data_transformation_config = config_manager.get_data_transformation_config()
        model_trainer_config = config_manager.get_model_trainer_config()
        model_evaluation_config = config_manager.get_model_evaluation_config()
    """

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> list:
        config_list = [
            self.config.data_ingestion_ravdess,
            self.config.data_ingestion_tess,
            self.config.data_ingestion_cremad,
            self.config.data_ingestion_savee,
        ]
        data_ingestion_config_list = []
        for config in config_list:
            create_directories([config.local_data_path])
            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                local_data_path=config.local_data_path,
            )
            data_ingestion_config_list.append(data_ingestion_config)
        return data_ingestion_config_list

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            unzip_ravdess_dir=config.unzip_ravdess_dir,
            unzip_tess_dir=config.unzip_tess_dir,
            unzip_cremad_dir=config.unzip_cremad_dir,
            unzip_savee_dir=config.unzip_savee_dir,
            local_output_path=config.local_output_path,
            validation_status=config.validation_status,
            metadata_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            metadata_path=config.metadata_path,
            output_path=config.output_path,
            train_path=config.train_path,
            test_path=config.test_path,
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_params
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            model_params=params,
            target_col=schema.name,
        )

        return model_trainer_config

    def get_model_evaluation_config(self):
        config = self.config.model_evaluation
        schema = self.schema.TARGET_COLUMN
        params = self.params.model_parans

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            model_params=params,
            metric_file_name=config.metric_file_name,
            target_col=schema.name,
            mlflow_uri=config.mlflow_uri,
        )

        return model_evaluation_config
