import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import optuna
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import seaborn as sns
import matplotlib.pyplot as plt


def cnn_model_2(input_shape, n_filters, dropout_rate):
    """
    Creates a CNN model for speech emotion recognition.

    Args:
        input_shape (tuple): The input shape of the model (height, width, channels).
        n_filters (int): The number of filters in convolutional layers.
        dropout_rate (float): The dropout rate for regularization.

    Returns:
        keras.models.Sequential: The created CNN model.
    """
    model = Sequential()
    model.add(Conv1D(n_filters, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(n_filters * 2, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(n_filters * 4, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(7, activation="softmax"))  # 7 classes for emotions

    return model


def hp_tune_cnn(trial, X_train, y_train_enc):
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
    n_filters = trial.suggest_int("n_filters", 32, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    # Build the CNN model with the suggested hyperparameters
    input_shape = (X_train.shape[1], 1)
    model = cnn_model_2(input_shape, n_filters, dropout_rate)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train_enc, epochs=2, batch_size=64, verbose=0)

    # Evaluate the model
    _, accuracy = model.evaluate(X_train, y_train_enc, verbose=0)

    return accuracy


# Load and preprocess data
# Assuming you have a DataFrame 'Emotions' containing your data
X = Emotions.iloc[:, :-1].values
Y = Emotions["Emotions"].values
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=42, test_size=0.2, shuffle=True
)

# Perform hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: hp_tune_cnn(trial, X_train, y_train), n_trials=5)

# Get best hyperparameters
best_params = study.best_params

# Train model with best hyperparameters
best_model = cnn_model_2(
    (X_train.shape[1], 1), best_params["n_filters"], best_params["dropout_rate"]
)
best_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
best_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Evaluate model
evaluation_results = best_model.evaluate(X_test, y_test, verbose=0)
print("Evaluation Metrics:")
print(f"Accuracy: {evaluation_results[1]}")

# Calculate additional evaluation metrics
y_pred = best_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
precision = precision_score(y_true_labels, y_pred_labels, average="weighted")
recall = recall_score(y_true_labels, y_pred_labels, average="weighted")
f1 = f1_score(y_true_labels, y_pred_labels, average="weighted")
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
class_report = classification_report(y_true_labels, y_pred_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.categories_[0],
    yticklabels=encoder.categories_[0],
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
