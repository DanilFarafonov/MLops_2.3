import mlflow
from mlflow.tracking import MlflowClient
import os


def test_model(model, x_test, y_test):
    os.environ["MLFLOW_REGISTRY_URI"] = "/home/xflow/project/mlflow/"
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("model_testing")

    with mlflow.start_run():
        score = model.evaluate(x_test, y_test)

        mlflow.log_artifact(local_path="/home/xflow/project/scripts/model_testing.py",
                            artifact_path="model_testing code")
        mlflow.end_run()

    return score
