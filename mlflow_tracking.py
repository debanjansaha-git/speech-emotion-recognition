import os
from random import random, randint

import mlflow

class MLFlowTracking:
    def __init__(self):
        mlflow.set_tracking_uri("http://mlflow:5000")

    def run_experiment(self):
        with mlflow.start_run():
            print("Running mlflow_tracking.py")

            mlflow.log_param("param1", randint(0, 100))

            mlflow.log_metric("foo", random())
            mlflow.log_metric("foo", random() + 1)
            mlflow.log_metric("foo", random() + 2)

            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            with open("outputs/test.txt", "w") as f:
                f.write("hello world!")

            mlflow.log_artifacts("outputs")

    def custom_function(self):
        # Define your custom function here
        print("Running custom function")

    def log_params(self, params):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

if __name__ == "__main__":
    tracker = MLFlowTracking()
    tracker.run_experiment()

    custom_params = {"param2": 0.5, "param3": "example"}
    tracker.log_params(custom_params)

    custom_metrics = {"bar": 42, "baz": 3.14}
    tracker.log_metrics(custom_metrics)

    tracker.custom_function()
