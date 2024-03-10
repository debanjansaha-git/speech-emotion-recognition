from mlcore.config.configuration import ConfigurationManager
from mlcore.components.data_validation import DataValidation
from mlcore import logger

STAGE_NAME = "data validation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            data_validataion_config = config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validataion_config)
            data_validation.generate_metadata()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
