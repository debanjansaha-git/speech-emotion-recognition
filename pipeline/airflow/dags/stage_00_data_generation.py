from mlcore.config.configuration import ConfigurationManager
from mlcore.components.data_generation import DataGeneration
from mlcore import logger

STAGE_NAME = "data generation stage"


class DataGenerationTrainingPipeline:
    """
    Class for the data ingestion training pipeline.

    Summary:
        This class represents the data ingestion training pipeline.

    Explanation:
        The DataIngestionTrainingPipeline class provides a main method to execute the data ingestion process.
        It initializes the ConfigurationManager and retrieves the data ingestion configuration.
        It then iterates over the data ingestion configurations and performs data ingestion by calling the DataIngestion class.

    Methods:
        main():
            Executes the data ingestion training pipeline by initializing the ConfigurationManager and performing data ingestion.

    Raises:
        Any exceptions that occur during the data ingestion process.

    Examples:
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_generation_config()
            data_ingestion = DataGeneration(config=data_ingestion_config)
            data_ingestion.load_1000_files()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = DataGenerationTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
