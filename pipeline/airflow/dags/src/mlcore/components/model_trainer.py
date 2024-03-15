import os
import yaml
import timeit
import joblib
import numpy as np
import pandas as pd

import optuna
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback

from mlcore.constants import *
from mlcore import logger
from mlcore.entity.config_entity import ModelTrainerConfig


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    Approved by the legend himself, François Chollet, creator of Keras!
    See thread: https://github.com/keras-team/keras/pull/5059
    """

    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        msg = "{Epoch: %i} %s" % (
            epoch,
            ", ".join("%s: %f" % (k, v) for k, v in logs.items()),
        )
        self.print_fcn(msg)


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
        self.model_params = model_params["model_params"]["CNN"]
        self.encoder = OneHotEncoder()

    def hp_tune_cnn(self, trial, X_train, y_train_enc):
        """
        Performs hyperparameter tuning for the CNN model using Optuna.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            X_train (numpy.ndarray): The training data.
            y_train_enc (numpy.ndarray): The encoded training labels.

        Returns:
            float: The mean accuracy score obtained during hyperparameter tuning.
        """
        # Define the hyperparameters to be tuned
        n_filters = trial.suggest_int("n_filters", 64, 256, log=True)
        kernel_size = trial.suggest_int("kernel_size", 3, 10)
        pool_size = trial.suggest_int("pool_size", 2, 5)
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)

        # Build the CNN model with the suggested hyperparameters
        model = self.cnn_model_1(
            X_train.shape[1], n_filters, kernel_size, pool_size, dropout_rate
        )
        model.compile(
            optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        # Train the model with cross-validation
        accuracy_scores = []
        for _ in range(3):  # Perform 3-fold cross-validation
            model.fit(X_train, y_train_enc, epochs=2, batch_size=64, verbose=0)
            _, accuracy = model.evaluate(X_train, y_train_enc, verbose=0)
            accuracy_scores.append(accuracy)

        # Compute the mean accuracy score
        accuracy = np.mean(accuracy_scores)
        return accuracy

    def prep_data_for_training(self):
        """
        Prepares the data for model training by performing necessary preprocessing steps.

        Returns:
            tuple: A tuple containing the preprocessed training and validation data.

        Examples:
            X_train, y_train_enc, X_val, y_val_enc = prep_data_for_training()
        """
        train_data = pd.read_parquet(self.config.train_path)
        val_data = pd.read_parquet(self.config.val_path)
        X_train = train_data.drop("Emotions", axis=1)
        y_train = train_data["Emotions"]
        X_val = val_data.drop("Emotions", axis=1)
        y_val = val_data["Emotions"]
        y_train_enc = self.encoder.fit_transform(
            np.array(y_train).reshape(-1, 1)
        ).toarray()
        y_val_enc = self.encoder.fit_transform(np.array(y_val).reshape(-1, 1)).toarray()
        X_train = np.expand_dims(X_train, axis=2)
        X_val = np.expand_dims(X_val, axis=2)

        return X_train, y_train_enc, X_val, y_val_enc

    def train(self, hypertune=False):
        """
        Trains the model using the prepared training and validation data.

        Args:
            hypertune (bool, optional): Whether to perform hyperparameter tuning using Optuna. Defaults to False.

        Examples:
            # Example 1: Train the model without hyperparameter tuning: train()
            # Example 2: Train the model with hyperparameter tuning:    train(hypertune=True)
        """
        best_params = self.model_params
        X_train, y_train_enc, X_val, y_val_enc = self.prep_data_for_training()
        logger.info("Archiving train-val datasets to disk...")
        np.save(f"{self.config.root_dir}/X_train.npy", X_train)
        np.save(f"{self.config.root_dir}/X_val.npy", X_val)
        np.save(f"{self.config.root_dir}/y_train.npy", y_train_enc)
        np.save(f"{self.config.root_dir}/y_val.npy", y_val_enc)
        logger.info(f"Train Data: {X_train.shape}, Train Targets: {y_train_enc.shape}")

        # Hyper Parameter Tuning
        if hypertune:
            logger.info("=== Hyperparameter Tuning using Optuna ===")
            study = optuna.create_study(direction="maximize")
            # study.optimize(self.hp_tune, (n_trials=25, x_train, y_train))
            study.optimize(
                lambda trial: self.hp_tune_cnn(trial, X_train, y_train_enc),
                n_trials=5,
            )
            best_params = study.best_params
            logger.info(f"Best Parameters Found: {best_params}")

            # Update best hyperparameters
            self.model_params = best_params
            # # Write new hyperparameters
            # if self.model_params != best_params:
            #     with open(PARAMS_FILE_PATH, "r") as f:
            #         tuned_params = yaml.safe_load(f)
            #     tuned_params["model_params"]["CNN_1"] = best_params
            #     with open(PARAMS_FILE_PATH, "w") as f:
            #         yaml.dump(tuned_params, f, default_flow_style=False)

        # Create a CNN model
        model = self.cnn_model_1(X_train.shape[1], **self.model_params)
        model.compile(
            optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.summary(print_fn=logger.info)
        logger.info("Begin Model Training")
        start = timeit.default_timer()
        rlrp = ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, verbose=0, patience=2, min_lr=0.0000001
        )
        history = model.fit(
            X_train,
            y_train_enc,
            batch_size=64,
            epochs=2,
            validation_data=(X_val, y_val_enc),
            callbacks=[rlrp, LoggingCallback(logger.info)],
        )
        elapsed_time = timeit.default_timer() - start
        logger.info(f"Training Duration: {elapsed_time:.2f} secs")

        # Save model
        logger.info("Export Trained Model for future inference")
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))

    def cnn_model_1(
        self, inp_shape, n_filters, kernel_size, pool_size, dropout_rate, **kwargs
    ):
        """
        Creates a CNN model for speech emotion recognition.

        Args:
            inp_shape (int):        The input shape of the model.
            n_filters (int):        The number of filters in the convolutional layers.
            kernel_size (int):      The size of the convolutional kernel.
            pool_size (int):        The size of the max pooling window.
            dropout_rate (float):   The dropout rate for regularization.
            **kwargs:               Additional keyword arguments.

        Returns:
            tensorflow.keras.models.Sequential: The created CNN model.

        Examples:
            # Example 1: Create a CNN model
            model = cnn_model_1(inp_shape=100, n_filters=32, kernel_size=3, pool_size=2, dropout_rate=0.2)
        """
        model = Sequential()
        model.add(
            Conv1D(
                n_filters,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                activation="relu",
                input_shape=(inp_shape, 1),
            )
        )
        model.add(MaxPooling1D(pool_size=pool_size, strides=2, padding="same"))

        model.add(
            Conv1D(
                n_filters,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                activation="relu",
            )
        )
        model.add(MaxPooling1D(pool_size=pool_size, strides=2, padding="same"))

        model.add(
            Conv1D(
                n_filters // 2,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                activation="relu",
            )
        )
        model.add(MaxPooling1D(pool_size=pool_size, strides=2, padding="same"))
        model.add(Dropout(dropout_rate))

        model.add(
            Conv1D(
                n_filters // 4,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                activation="relu",
            )
        )
        model.add(MaxPooling1D(pool_size=pool_size, strides=2, padding="same"))
        model.add(Flatten())

        model.add(Dense(units=32, activation="relu"))
        model.add(Dropout(dropout_rate))

        model.add(Dense(units=7, activation="softmax"))

        return model
