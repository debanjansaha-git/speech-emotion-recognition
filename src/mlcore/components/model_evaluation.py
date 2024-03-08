from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import numpy as np
import joblib
from mlcore.utils.common import save_json
from pathlib import Path
from mlcore.entity.config_entity import ModelEvaluationConfig
from mlcore.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self, ytrue, ypred):
        rmse = np.sqrt(mean_squared_error(ytrue, ypred))
        mae = mean_absolute_error(ytrue, ypred)
        r2 = r2_score(ytrue, ypred)

        train_data = pd.read_csv(self.config.train_data_path)
        xtrain = np.array(train_data["Open"]).reshape(-1, 1)
        ytrain = np.array(train_data[[self.config.target_col]]).reshape(-1)

        model = joblib.load(self.config.model_path)

        cv_score = cross_val_score(model, xtrain, ytrain, cv=5)
        return rmse, mae, r2, cv_score

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        xtest = np.array(test_data["Open"]).reshape(-1, 1)
        ytest = np.array(test_data[[self.config.target_col]]).reshape(-1)

        os.environ["MLFLOW_TRACKING_URI"] = (
            "http://ec2-15-206-173-174.ap-south-1.compute.amazonaws.com:5000/"
        )
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            ypred = model.predict(xtest)

            (rmse, mae, r2, cv_score) = self.evaluate_model(ytest, ypred)

            cv_score = cv_score.tolist()

            scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mean_cv_accuracy": np.mean(cv_score),
                "cv_accuracy_std": np.std(cv_score),
            }

            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mean_cv_accuracy", np.mean(cv_score))
            mlflow.log_metric("cv_accuracy_std", np.std(cv_score))

            # if tracking_url_type_store != "file":
            #     mlflow.sklearn.log_model(model, "model", registered_model_name = "RandomForestRegressor")
            # else:
            mlflow.sklearn.log_model(model, "model")
