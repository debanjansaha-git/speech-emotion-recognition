from mlcore.config.configuration import ConfigurationManager
from mlcore.components.data_ingestion import DataIngestion
from mlcore import logger

STAGE_NAME = "data ingestion stage"


class DataIngestionTrainingPipeline:
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
            data_ingestion_config_list = config_manager.get_data_ingestion_config()
            for data_ingestion_config in data_ingestion_config_list:
                data_ingestion = DataIngestion(config=data_ingestion_config)
                # data_ingestion.download_data()
                data_ingestion.download_from_gcp()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
