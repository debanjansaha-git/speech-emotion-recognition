import pandas as pd
import os
from mlcore import logger
from sklearn.ensemble import RandomForestClassifier
import optuna
import yaml
import joblib
import numpy as np
from mlcore.entity.config_entity import ModelTrainerConfig
from mlcore.constants import *
from sklearn.model_selection import cross_val_score
import timeit


class ModelTrainer:
    """
    Class for model training.

    Summary:
        This class handles the training of a Random Forest model using the specified configuration.

    Explanation:
        The ModelTrainer class provides methods to train a Random Forest model.
        The class takes a ModelTrainerConfig object as input, which contains the necessary configuration parameters for model training.
        The hp_tune() method performs hyperparameter tuning using Optuna to find the best set of hyperparameters for the model.
        The train() method trains the Random Forest model using the specified hyperparameters and saves the trained model to disk.

    Args:
        config (ModelTrainerConfig): The configuration object containing the necessary parameters for model training.

    Methods:
        hp_tune(trial: optuna.Trial, xtrain: np.ndarray, ytrain: np.ndarray) -> float:
            Performs hyperparameter tuning using Optuna and returns the accuracy score.

        train(hypertune: bool = True):
            Trains the Random Forest model using the specified hyperparameters and saves the trained model to disk.

    Raises:
        No transformation parameters specified: If no transformation parameters are specified in the configuration.

    Examples:
        model_trainer = ModelTrainer(config)
        accuracy = model_trainer.hp_tune(trial, x_train, y_train)
        model_trainer.train(hypertune=True)
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        with open(PARAMS_FILE_PATH, "r") as f:
            model_params = yaml.safe_load(f)
        self.model_params = model_params["model_params"]["RandomForestClassifier"]

    def hp_tune(self, trial, xtrain, ytrain):
        # crit = trial.suggest_categorical("criterion", ["entropy"])
        n_est = trial.suggest_int("n_estimators", 2, 200, log=True)
        m_depth = trial.suggest_int("max_depth", 1, 100, log=True)
        rfc = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth)

        score = cross_val_score(rfc, xtrain, ytrain, cv=3)
        accuracy = score.mean()
        return accuracy

    def train(self, hypertune=True):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        best_params = self.params.model_parans

        x_train = np.array(train_data.drop([self.config.target_col], axis=1))
        x_test = np.array(test_data.drop([self.config.target_col], axis=1))
        y_train = np.array(train_data[[self.config.target_col]])
        y_test = np.array(test_data[[self.config.target_col]])
        logger.info("Archiving train-test datasets to disk...")
        np.save(f"{self.config.root_dir}/X_train.npy", x_train)
        np.save(f"{self.config.root_dir}/X_test.npy", x_test)
        np.save(f"{self.config.root_dir}/y_train.npy", y_train)
        np.save(f"{self.config.root_dir}/y_test.npy", y_test)
        logger.info(f"Train Data: {x_train.shape}, Test Data: {y_train.shape}")

        # Hyper Parameter Tuning
        if hypertune:
            logger.info("=== Hyperparameter Tuning using Optuna ===")
            study = optuna.create_study(direction="maximize")
            # study.optimize(self.hp_tune, (n_trials=25, x_train, y_train))
            study.optimize(
                lambda trial: self.hp_tune(trial, x_train, y_train), n_trials=25
            )
            best_params = study.best_params
            logger.info(f"Best Parameters Found: {best_params}")

            # Write new hyperparameters
            if self.model_params != best_params:
                with open(PARAMS_FILE_PATH, "r") as f:
                    tuned_params = yaml.safe_load(f)
                tuned_params["model_params"]["RandomForestClassifier"] = best_params
                with open(PARAMS_FILE_PATH, "w") as f:
                    yaml.dump(tuned_params, f, default_flow_style=False)

        # Create a Random Forest Model
        rfc = RandomForestClassifier(self.model_params)
        logger.info("Begin Model Training")
        start = timeit.default_timer()
        rfc.fit(x_train, y_train)
        elapsed_time = timeit.default_timer() - start
        logger.info(f"Training Duration: {elapsed_time:.2f} secs")

        # Save model
        logger.info("Export Trained Model for future inference")
        joblib.dump(rfc, os.path.join(self.config.root_dir, self.config.model_name))
