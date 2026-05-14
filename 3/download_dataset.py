import os
import numpy as np
from torchvision import datasets, transforms
from scripts.utils.paths import LOCAL_DATA_DIR


def download_dataset() -> None:
    raw_dir = os.path.join(LOCAL_DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=raw_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=raw_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_images = train_dataset.data.numpy().astype("float32") / 255.0
    train_labels = train_dataset.targets.numpy().astype("int64")

    test_images = test_dataset.data.numpy().astype("float32") / 255.0
    test_labels = test_dataset.targets.numpy().astype("int64")

    np.savez_compressed(
        os.path.join(raw_dir, "train.npz"),
        images=train_images,
        labels=train_labels,
    )
    np.savez_compressed(
        os.path.join(raw_dir, "test.npz"),
        images=test_images,
        labels=test_labels,
    )


if __name__ == "__main__":
    download_dataset()


