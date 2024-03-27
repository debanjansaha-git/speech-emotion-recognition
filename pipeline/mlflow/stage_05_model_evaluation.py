from mlcore.config.configuration import ConfigurationManager
from mlcore.components.model_evaluation import ModelEvaluation
from mlcore import logger

STAGE_NAME = "model evaluation stage"


class ModelEvaluationTrainingPipeline:
    """
    Class for model evaluation training pipeline.

    Summary:
        This class represents the model evaluation training pipeline.

    Explanation:
        The ModelEvaluationTrainingPipeline class provides a main method to execute the model evaluation training pipeline.
        It initializes the ConfigurationManager and retrieves the model evaluation configuration.
        It then performs model evaluation by calling the ModelEvaluation class.

    Methods:
        main():
            Executes the model evaluation training pipeline by initializing the ConfigurationManager and performing model evaluation.

    Raises:
        Any exceptions that occur during the model evaluation training pipeline.

    Examples:
        pipeline = ModelEvaluationTrainingPipeline()
        pipeline.main()
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            model_eval_config = config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_eval_config)
            model_evaluation.log_into_mlflow()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
