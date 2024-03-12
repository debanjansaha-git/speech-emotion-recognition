from src.mlcore import logger
from mlcore.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from mlcore.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)
from mlcore.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)
from mlcore.pipeline.stage_04_model_trainer import (
    ModelTrainerTrainingPipeline,
)
from mlcore.pipeline.stage_05_model_evaluation import (
    ModelEvaluationTrainingPipeline,
)

STAGE_NAME = "data ingestion stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "data validation stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_val = DataValidationTrainingPipeline()
    data_val.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
except Exception as e:
    logger.exception(e)
    raise e


# STAGE_NAME = "data transformation stage"
# try:
#     logger.info(f">>>> stage {STAGE_NAME} started <<<<")
#     data_transform = DataTransformationTrainingPipeline()
#     data_transform.main()
#     logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Model training stage"
# try:
#     logger.info(f">>>> stage {STAGE_NAME} started <<<<")
#     model_train = ModelTrainerTrainingPipeline()
#     model_train.main()
#     logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Model evaluation stage"
# try:
#     logger.info(f">>>> stage {STAGE_NAME} started <<<<")
#     model_eval = ModelEvaluationTrainingPipeline()
#     model_eval.main()
#     logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
# except Exception as e:
#     logger.exception(e)
#     raise e
