from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from scripts.utils.paths import LOCAL_DATA_DIR, S3_BUCKET, TRAIN_DATA_PREFIX, TEST_DATA_PREFIX


def download_dataset_task():
    from scripts.data.download_dataset import download_dataset
    download_dataset()

def create_bucket_task():
    from scripts.data.create_bucket import create_bucket
    create_bucket(S3_BUCKET)

def upload_train_data_task():
    from scripts.data.upload_to_s3 import upload_train_data_to_s3
    upload_train_data_to_s3()


def upload_test_data_task():
    from scripts.data.upload_to_s3 import upload_test_data_to_s3
    upload_test_data_to_s3()


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "data_generator_dag",
    default_args=default_args,
    description="DAG для генерации и загрузки данных на S3",
    schedule_interval=None,
    catchup=False,
    tags=["data", "s3"],
)

download_data_op = PythonOperator(
    task_id="download_data_op",
    python_callable=download_dataset_task,
    dag=dag,
)

create_bucket_op = PythonOperator(
    task_id="create_bucket_op",
    python_callable=create_bucket_task,
    dag=dag,
)

upload_train_data_op = PythonOperator(
    task_id="upload_train_data_op",
    python_callable=upload_train_data_task,
    dag=dag,
)

upload_test_data_op = PythonOperator(
    task_id="upload_test_data_op",
    python_callable=upload_test_data_task,
    dag=dag,
)

download_data_op >> create_bucket_op >> [upload_train_data_op, upload_test_data_op]