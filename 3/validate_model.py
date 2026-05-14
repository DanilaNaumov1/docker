import json
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score
from scripts.utils.model_utils import load_model
from scripts.utils.paths import LOCAL_DATA_DIR, LOCAL_MODELS_DIR, MODEL_PREFIX
from scripts.utils.mlflow_utils import log_model_and_metrics


def validate_model() -> None:

    model_path = os.path.join(LOCAL_MODELS_DIR, "model.pkl")
    model = load_model(model_path)

    val_path = os.path.join(LOCAL_DATA_DIR, "processed", "val.npz")
    val_data = np.load(val_path)
    X_val = val_data["images"]
    y_val = val_data["labels"]

    n_samples = X_val.shape[0]
    X_val_flat = X_val.reshape(n_samples, -1)

    y_pred = model.predict(X_val_flat)

    accuracy = accuracy_score(y_val, y_pred)

    s3_model_key = f"{MODEL_PREFIX}model.pkl"

    metrics = {
        "accuracy": float(accuracy),
        "n_samples": int(n_samples),
        "s3_model_key": s3_model_key,
        "validated_at": datetime.utcnow().isoformat(),
    }

    metrics_path = os.path.join(LOCAL_MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Метрики сохранены в {metrics_path}")
    print(f"Accuracy: {accuracy:.4f}")

    params = {
        "model_type": "RandomForestClassifier",
        "s3_model_key": s3_model_key,
    }
    log_model_and_metrics(model, accuracy=accuracy, params=params)


if __name__ == "__main__":
    validate_model()

