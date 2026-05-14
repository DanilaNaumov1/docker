import os
import numpy as np
from sklearn.model_selection import train_test_split
from scripts.utils.paths import LOCAL_DATA_DIR


def split_data(test_size: float = 0.2, random_state: int = 42) -> None:
    raw_dir = os.path.join(LOCAL_DATA_DIR, "raw")
    processed_dir = os.path.join(LOCAL_DATA_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(raw_dir, "train.npz")
    test_path = os.path.join(raw_dir, "test.npz")

    train_data = np.load(train_path)
    X = train_data["images"]
    y = train_data["labels"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    np.savez_compressed(
        os.path.join(processed_dir, "train.npz"),
        images=X_train,
        labels=y_train,
    )
    np.savez_compressed(
        os.path.join(processed_dir, "val.npz"),
        images=X_val,
        labels=y_val,
    )

    test_data = np.load(test_path)
    np.savez_compressed(
        os.path.join(processed_dir, "test.npz"),
        images=test_data["images"],
        labels=test_data["labels"],
    )


if __name__ == "__main__":
    split_data()


