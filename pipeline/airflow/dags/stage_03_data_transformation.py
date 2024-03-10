from mlcore.config.configuration import ConfigurationManager
from mlcore.components.data_transformation import DataTransformation
from mlcore import logger
from pathlib import Path

STAGE_NAME = "data transformation stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(
                Path("src/mlcore/artifacts/data_validation/status.txt"), "r"
            ) as f:
                status = f.read().split(" ")[-1]
            if status == "True":
                config_manager = ConfigurationManager()
                data_transformation_config = (
                    config_manager.get_data_transformation_config()
                )
                data_transformation = DataTransformation(
                    config=data_transformation_config
                )
                data_transformation.feature_engineering()
                data_transformation.train_test_split_data(test_size=0.2)
            else:
                raise Exception("Data schema is not valid")
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
