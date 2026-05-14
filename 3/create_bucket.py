import boto3
from botocore.exceptions import ClientError
import os


def create_bucket(bucket_name: str) -> None:
    if os.environ.get('AWS_ACCESS_KEY_ID') == 'test-key':
        print("Используются тестовые AWS ключи - создаём локальные директории")
        base_dir = "/opt/airflow/data/s3"
        directories = [
            f"{base_dir}/data/train",
            f"{base_dir}/data/test",
            f"{base_dir}/models",
            f"{base_dir}/metrics",
            f"{base_dir}/inference",
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Создана директория: {directory}")
        return

    s3 = boto3.client("s3")

    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"Бакет {bucket_name} уже существует")
        return
    except ClientError:
        print(f"Создаём бакет {bucket_name}")
        pass

    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Бакет {bucket_name} успешно создан")
    except ClientError as e:
        print(f"Ошибка при создании бакета: {e}")
        if os.environ.get('AWS_ACCESS_KEY_ID') == 'test-key':
            print("Продолжаем в тестовом режиме...")


if __name__ == "__main__":
    from scripts.utils.paths import S3_BUCKET

    create_bucket(S3_BUCKET)