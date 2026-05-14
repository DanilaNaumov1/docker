import pickle
from typing import Any


def save_model(model: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> Any:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


