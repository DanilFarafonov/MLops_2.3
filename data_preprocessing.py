import pandas as pd
import keras
import mlflow
from mlflow.tracking import MlflowClient
import os


def preprocess(df_x_train, df_y_train, df_x_test, df_y_test):
    os.environ["MLFLOW_REGISTRY_URI"] = "/home/xflow/project/mlflow/"
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("data_preprocessing")

    df_x_train = pd.read_csv(df_x_train)
    df_y_train = pd.read_csv(df_y_train)
    df_x_test = pd.read_csv(df_x_test)
    df_y_test = pd.read_csv(df_y_test)

    with mlflow.start_run():
        x_train = df_x_train.to_numpy()
        y_train = df_y_train.to_numpy()
        x_test = df_x_test.to_numpy()
        y_test = df_y_test.to_numpy()

        x_train = x_train / 255
        x_test = x_test / 255

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        df_x_train = pd.DataFrame(x_train)
        df_y_train = pd.DataFrame(y_train)
        df_x_test = pd.DataFrame(x_test)
        df_y_test = pd.DataFrame(y_test)

        mlflow.log_artifact(local_path="/home/xflow/project/scripts/data_preprocessing.py",
                            artifact_path="preprocess_data code")
        mlflow.end_run()

    df_x_train.to_csv("train/x_train_preprocessing.csv", index=False)
    df_y_train.to_csv("train/y_train_preprocessing.csv", index=False)
    df_x_test.to_csv("test/x_test_preprocessing.csv", index=False)
    df_y_test.to_csv("test/y_test_preprocessing.csv", index=False)

    return "train/x_train_preprocessing.csv", "train/y_train_preprocessing.csv", \
           "test/x_test_preprocessing.csv", "test/y_test_preprocessing.csv"
