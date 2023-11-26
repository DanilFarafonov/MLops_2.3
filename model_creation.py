from keras.models import Sequential
from keras.layers import Dense
import mlflow
from mlflow.tracking import MlflowClient
import os


def create_model(x_train, y_train):
    os.environ["MLFLOW_REGISTRY_URI"] = "/home/xflow/project/mlflow/"
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("model_creation")

    with mlflow.start_run():
        model = Sequential()
        model.add(Dense(800, input_dim=784, activation="relu"))
        model.add(Dense(10, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train,
                  batch_size=200,
                  epochs=50,
                  verbose=1)

        mlflow.log_artifact(local_path="/home/xflow/project/scripts/model_creation.py",
                            artifact_path="model_creation code")
        mlflow.end_run()

    return model
