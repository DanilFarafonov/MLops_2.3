import keras
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os


def load():
    os.environ["MLFLOW_REGISTRY_URI"] = "/home/xflow/project/mlflow/"
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("data_load")

    with mlflow.start_run():
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        df_x_train = pd.DataFrame(x_train.reshape(60000, 784))
        df_y_train = pd.DataFrame(y_train)
        df_x_test = pd.DataFrame(x_test.reshape(10000, 784))
        df_y_test = pd.DataFrame(y_test)
        mlflow.log_artifact(local_path="/home/xflow/project/scripts/data_loading.py",
                            artifact_path="load_data code")
        mlflow.end_run()

    return df_x_train, df_y_train, df_x_test, df_y_test
