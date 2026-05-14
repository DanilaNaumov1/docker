import os
import numpy as np
import pandas as pd
from scripts.utils.model_utils import load_model
from scripts.utils.paths import LOCAL_DATA_DIR, LOCAL_MODELS_DIR, LOCAL_RESULTS_DIR


def run_inference() -> None:

    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(LOCAL_MODELS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    model = load_model(model_path)

    test_path = os.path.join(LOCAL_DATA_DIR, "processed", "test.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test данные не найдены: {test_path}")
    
    test_data = np.load(test_path)
    X_test = test_data["images"]
    y_test = test_data["labels"]

    n_samples = X_test.shape[0]
    X_test_flat = X_test.reshape(n_samples, -1)

    y_pred = model.predict(X_test_flat)

    results_df = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_pred,
    })

    results_path = os.path.join(LOCAL_RESULTS_DIR, "predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Результаты сохранены в {results_path}")
    print(f"Всего предсказаний: {len(y_pred)}")


if __name__ == "__main__":
    run_inference()

