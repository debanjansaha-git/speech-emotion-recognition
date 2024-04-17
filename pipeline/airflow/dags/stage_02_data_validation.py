from mlcore.config.configuration import ConfigurationManager
from mlcore.components.data_validation import DataValidation
from mlcore import logger

STAGE_NAME = "data validation stage"


class DataValidationTrainingPipeline:
    """
    Class for data validation.

    Summary:
        This class represents the data validation process.

    Explanation:
        The DataValidation class provides a placeholder for the data validation process.
        It does not have any specific behavior or functionality.

    Methods:
        None.

    Raises:
        None.

    Examples:
        validation = DataValidation()
        # No specific usage examples for this class.
    """

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
