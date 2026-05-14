from datetime import datetime, timedelta
from airflow import DAG
from airflow.sensors.python import PythonSensor
from airflow.operators.python import PythonVirtualenvOperator
from scripts.utils.paths import *

def check_for_train_data():
    from scripts.utils.s3_utils import check_exists
    return check_exists(S3_BUCKET, f"{TRAIN_DATA_PREFIX}train.npz")

default_args = {
    "owner": "student_kapralov", 
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    "model_training_workflow", 
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["hw_mlops", "training"],
) as dag:

    wait_for_data = PythonSensor(
        task_id="data_sensor",
        python_callable=check_for_train_data,
        timeout=600,
        poke_interval=30,
    )

    def download_data_func():
        from scripts.utils.s3_utils import download_file_from_s3
        import os
        os.makedirs(f"{LOCAL_DATA_DIR}/raw", exist_ok=True)
        download_file_from_s3(S3_BUCKET, f"{TRAIN_DATA_PREFIX}train.npz", f"{LOCAL_DATA_DIR}/raw/train.npz")

    get_data_op = PythonVirtualenvOperator(
        task_id="get_data_op",
        python_callable=download_data_func,
        requirements=["boto3"],
    )

    def split_step():
        from scripts.data.split_data import split_data
        print("Начинаем разделение данных на выборки...")
        split_data(test_size=0.15) 

    split_data_op = PythonVirtualenvOperator(
        task_id="split_data_op",
        python_callable=split_step,
        requirements=["numpy", "scikit-learn"],
    )

    def train_step():
        from scripts.training.train_model import train_model
        train_model()

    train_op = PythonVirtualenvOperator(
        task_id="train_op",
        python_callable=train_step,
        requirements=["scikit-learn", "numpy"],
    )

    def upload_model_step():
        import os
        from scripts.utils.s3_utils import upload_file_to_s3
        path = os.path.join("/opt/airflow/models", "model.pkl")
        upload_file_to_s3(path, "ml-bucket", "models/model.pkl")

    model_upload_op = PythonVirtualenvOperator(
        task_id="model_upload_op",
        python_callable=upload_model_step,
        requirements=["boto3"],
    )

    def eval_step():
        from scripts.training.validate_model import validate_model
        validate_model()

    evaluate_data_op = PythonVirtualenvOperator(
        task_id="evaluate_data_op",
        python_callable=eval_step,
        requirements=["scikit-learn", "numpy", "mlflow"],
    )

    wait_for_data >> get_data_op >> split_data_op >> train_op >> model_upload_op >> evaluate_data_op