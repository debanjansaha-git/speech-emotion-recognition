from mlcore.config.configuration import ConfigurationManager
from mlcore.components.model_trainer import ModelTrainer
from mlcore import logger
from pathlib import Path

STAGE_NAME = "model training stage"


class ModelTrainerTrainingPipeline:
    """
    Class for model trainer training pipeline.

    Summary:
        This class represents the model trainer training pipeline.

    Explanation:
        The ModelTrainerTrainingPipeline class provides a main method to execute the model trainer training pipeline.
        It initializes the ConfigurationManager and retrieves the model trainer configuration.
        It then performs model training by calling the ModelTrainer class.

    Methods:
        main():
            Executes the model trainer training pipeline by initializing the ConfigurationManager and performing model training.

    Raises:
        Any exceptions that occur during the model trainer training pipeline.

    Examples:
        pipeline = ModelTrainerTrainingPipeline()
        pipeline.main()
    """

    def __init__(self, hypertune=False, epochs=2):
        self.hypertune = hypertune
        self.epochs = epochs

    def main(self):
        try:
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train(hypertune=self.hypertune, epochs=self.epochs)
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
