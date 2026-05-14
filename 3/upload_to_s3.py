import os
from scripts.utils.paths import (
    LOCAL_DATA_DIR,
    S3_BUCKET,
    TRAIN_DATA_PREFIX,
    TEST_DATA_PREFIX,
)
from scripts.utils.s3_utils import upload_file_to_s3


def upload_train_data_to_s3() -> None:
    train_path = os.path.join(LOCAL_DATA_DIR, "raw", "train.npz")
    upload_file_to_s3(train_path, S3_BUCKET, f"{TRAIN_DATA_PREFIX}train.npz")
    print(f"Train данные загружены: {train_path} -> {TRAIN_DATA_PREFIX}train.npz")


def upload_test_data_to_s3() -> None:
    test_path = os.path.join(LOCAL_DATA_DIR, "raw", "test.npz")
    upload_file_to_s3(test_path, S3_BUCKET, f"{TEST_DATA_PREFIX}test.npz")
    print(f"Test данные загружены: {test_path} -> {TEST_DATA_PREFIX}test.npz")


if __name__ == "__main__":
    upload_train_data_to_s3()