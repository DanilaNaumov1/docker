import os
from typing import Any, Dict, Optional
import mlflow
import mlflow.sklearn


def _setup_mlflow() -> None:

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:/opt/airflow/mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "training_experiment")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_model_and_metrics(
    model: Any,
    accuracy: float,
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None:

    _setup_mlflow()

    with mlflow.start_run():
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        mlflow.log_metric("accuracy", float(accuracy))

        mlflow.sklearn.log_model(model, artifact_path="model")


