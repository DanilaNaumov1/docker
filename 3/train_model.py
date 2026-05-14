import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scripts.utils.model_utils import save_model
from scripts.utils.paths import LOCAL_DATA_DIR, LOCAL_MODELS_DIR

def train_model() -> None:
    print(f"--- Запуск обучения модели в директории: {LOCAL_MODELS_DIR} ---")
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

    data_path = os.path.join(LOCAL_DATA_DIR, "processed", "train.npz")
    with np.load(data_path) as data:
        X_train = data["images"]
        y_train = data["labels"]

    X_flat = X_train.reshape(X_train.shape[0], -1)

    clf = RandomForestClassifier(n_estimators=70, max_depth=10, random_state=13, n_jobs=-1)
    print("Обучение запущено...")
    clf.fit(X_flat, y_train)

    out_path = os.path.join(LOCAL_MODELS_DIR, "model.pkl")
    save_model(clf, out_path)
    print(f"Готово! Модель сохранена здесь: {out_path}")

if __name__ == "__main__":
    train_model()