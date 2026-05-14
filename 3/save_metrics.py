import os
from scripts.utils.paths import LOCAL_MODELS_DIR, S3_BUCKET, METRICS_PREFIX
from scripts.utils.s3_utils import upload_file_to_s3


def upload_metrics_to_s3() -> None:

    metrics_path = os.path.join(LOCAL_MODELS_DIR, "metrics.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Файл метрик не найден: {metrics_path}")

    s3_key = f"{METRICS_PREFIX}metrics.json"
    upload_file_to_s3(metrics_path, S3_BUCKET, s3_key)
    print(f"Метрики загружены в S3: s3://{S3_BUCKET}/{s3_key}")


if __name__ == "__main__":
    upload_metrics_to_s3()

