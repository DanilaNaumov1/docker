import os
from typing import Optional
import boto3
from botocore.exceptions import ClientError
import shutil


def _is_test_mode():
    return os.environ.get('AWS_ACCESS_KEY_ID') == 'test-key'


def _get_local_path(bucket: str, key: str) -> str:
    return f"/opt/airflow/data/s3/{key}"


def _get_s3_client():
    return boto3.client("s3")


def upload_file_to_s3(local_path: str, bucket: str, s3_key: str) -> None:

    if _is_test_mode():
        local_s3_path = _get_local_path(bucket, s3_key)
        os.makedirs(os.path.dirname(local_s3_path), exist_ok=True)
        shutil.copy2(local_path, local_s3_path)
        print(f"Файл скопирован локально: {local_path} -> {local_s3_path}")
        return

    s3 = _get_s3_client()
    s3.upload_file(Filename=local_path, Bucket=bucket, Key=s3_key)


def download_file_from_s3(bucket: str, s3_key: str, local_path: str) -> None:
    if _is_test_mode():
        local_s3_path = _get_local_path(bucket, s3_key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_s3_path):
            shutil.copy2(local_s3_path, local_path)
            print(f"Файл скопирован локально: {local_s3_path} -> {local_path}")
        else:
            raise FileNotFoundError(f"Файл не найден: {local_s3_path}")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3 = _get_s3_client()
    s3.download_file(Bucket=bucket, Key=s3_key, Filename=local_path)


def check_exists(bucket: str, key: str) -> bool:
    if _is_test_mode():
        local_s3_path = _get_local_path(bucket, key)
        return os.path.exists(local_s3_path)

    s3 = _get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code: Optional[str] = exc.response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def upload_directory_to_s3(local_dir: str, bucket: str, s3_prefix: str) -> None:

    if _is_test_mode():
        local_dir = os.path.abspath(local_dir)
        for root, _, files in os.walk(local_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, local_dir).replace("\\", "/")
                key = os.path.join(s3_prefix, rel_path).replace("\\", "/")
                upload_file_to_s3(full_path, bucket, key)
        return

    local_dir = os.path.abspath(local_dir)
    for root, _, files in os.walk(local_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, local_dir).replace("\\", "/")
            key = os.path.join(s3_prefix, rel_path).replace("\\", "/")
            upload_file_to_s3(full_path, bucket, key)


def download_directory_from_s3(bucket: str, s3_prefix: str, local_dir: str) -> None:

    if _is_test_mode():
        s3_base_path = _get_local_path(bucket, s3_prefix)
        if not os.path.exists(s3_base_path):
            return

        for root, _, files in os.walk(s3_base_path):
            for filename in files:
                s3_path = os.path.join(root, filename)
                rel_path = os.path.relpath(s3_path, s3_base_path).replace("\\", "/")
                local_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.copy2(s3_path, local_path)
        return

    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    os.makedirs(local_dir, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(s3_prefix):].lstrip("/").replace("\\", "/")
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(Bucket=bucket, Key=key, Filename=local_path)