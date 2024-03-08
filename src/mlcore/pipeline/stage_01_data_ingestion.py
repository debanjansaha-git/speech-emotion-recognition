from mlcore.config.configuration import ConfigurationManager
from mlcore.components.data_ingestion import DataIngestion
from mlcore import logger

STAGE_NAME = "data ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config_list = config_manager.get_data_ingestion_config()
            for data_ingestion_config in data_ingestion_config_list:
                data_ingestion = DataIngestion(config=data_ingestion_config)
                data_ingestion.download_data()
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
