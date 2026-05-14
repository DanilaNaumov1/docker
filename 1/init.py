import os
import time
import logging
import json
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StorageInit")

class ModelStorageManager:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "admin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "password123"),
            verify=False
        )

    def ensure_bucket(self, name: str):
        try:
            self.s3.head_bucket(Bucket=name)
        except ClientError:
            self.s3.create_bucket(Bucket=name)
            logger.info(f"Created bucket: {name}")

    def upload_model_assets(self, bucket: str):
        mapping = {
            "model.onnx": "codegen/1/model.onnx",
            "config.pbtxt": "codegen/config.pbtxt",
            "dali.config": "codegen_dali_ensemble/1/model.config",
            "preprocessor.config": "text_preprocessor/1/model.config",
            "postprocessor.config": "text_postprocessor/1/model.config"
        }
        
        for local, s3_key in mapping.items():
            if os.path.exists(local):
                self.s3.upload_file(local, bucket, s3_key)
                logger.info(f"Successfully deployed {local} to {s3_key}")
            else:
                logger.warning(f"Asset missing: {local}")

def main():
    manager = ModelStorageManager()
    
    for _ in range(10):
        try:
            manager.ensure_bucket(os.getenv("MODELS_BUCKET_NAME", "triton-models"))
            break
        except Exception:
            time.sleep(5)
            
    manager.upload_model_assets(os.getenv("MODELS_BUCKET_NAME", "triton-models"))

if __name__ == "__main__":
    main()