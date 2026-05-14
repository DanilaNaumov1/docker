import os
from typing import Optional
import mlflow


def choose_best_model() -> Optional[str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:/opt/airflow/mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "training_experiment")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Эксперимент MLflow '{experiment_name}' не найден")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError("В MLflow нет ни одного run с метрикой accuracy")

    best_run = runs.iloc[0]
    best_accuracy = best_run["metrics.accuracy"]
    best_model_key = best_run.get("params.s3_model_key")

    if not best_model_key:
        raise ValueError("У лучшего run в MLflow отсутствует параметр 's3_model_key'")

    print(
        f"Лучшая модель из MLflow: s3_key={best_model_key}, "
        f"accuracy={best_accuracy:.4f}, run_id={best_run['run_id']}"
    )
    return str(best_model_key)


if __name__ == "__main__":
    choose_best_model()

