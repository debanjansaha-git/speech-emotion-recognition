from mlcore.config.configuration import ConfigurationManager
from mlcore.components.data_transformation import DataTransformation
from mlcore import logger
from pathlib import Path

STAGE_NAME = "data transformation stage"


class DataTransformationTrainingPipeline:
    """
    Class for data transformation training pipeline.

    Summary:
        This class represents the data transformation training pipeline.

    Explanation:
        The DataTransformationTrainingPipeline class provides a main method to execute the data transformation training pipeline.
        It initializes the ConfigurationManager and retrieves the data transformation configuration.
        It then performs data transformation and model training by calling the DataTransformation and ModelTrainer classes.

    Methods:
        main():
            Executes the data transformation training pipeline by initializing the ConfigurationManager and performing data transformation and model training.

    Raises:
        Any exceptions that occur during the data transformation training pipeline.

    Examples:
        pipeline = DataTransformationTrainingPipeline()
        pipeline.main()
    """

    def __init__(self, stage='train'):
        self.stage = stage

    def main(self):
        try:
            # with open(
            #     Path("src/mlcore/artifacts/data_validation/status.txt"), "r"
            # ) as f:
            #     status = f.read().split(" ")[-1]
            # if status != "True":
            #     raise Exception("Data schema is not valid")
            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config, stage=self.stage)
            data_transformation.feature_engineering()
            data_transformation.scale_data()
            # data_transformation.split_and_scale(
            #     test_size=0.2, val_size=0.2, method="standard"
            # )
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
