from datetime import datetime
from airflow import DAG
from airflow.sensors.python import PythonSensor
from airflow.operators.python import PythonOperator
from scripts.utils.paths import *

def check_s3_files():
    from scripts.utils.s3_utils import check_exists
    m_ok = check_exists(S3_BUCKET, f"{MODEL_PREFIX}model.pkl")
    d_ok = check_exists(S3_BUCKET, f"{TEST_DATA_PREFIX}test.npz")
    return m_ok and d_ok

default_args = {
    "owner": "student_kapralov",
    "start_date": datetime(2024, 1, 1),
}

dag = DAG(
    "model_inference_flow",
    default_args=default_args,
    schedule_interval="@daily", 
    catchup=False
)

wait_all_files = PythonSensor(
    task_id="wait_for_files",
    python_callable=check_s3_files,
    dag=dag
)

def choose_and_download_model(**kwargs):
    from scripts.inference.choose_best_model import choose_best_model
    from scripts.utils.s3_utils import download_file_from_s3
    import os
    
    print("Выбираем лучшую версию модели через MLflow...")
    best_key = choose_best_model()
    
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    target = os.path.join(LOCAL_MODELS_DIR, "model.pkl")
    download_file_from_s3(S3_BUCKET, best_key, target)
    print(f"Модель {best_key} успешно скачана.")

choose_model_op = PythonOperator(
    task_id="choose_and_download_model_op",
    python_callable=choose_and_download_model,
    provide_context=True,
    dag=dag
)

def run_predictions():
    from scripts.inference.run_inference import run_inference
    print("Запуск предсказаний на тестовых данных...")
    run_inference()

inference_op = PythonOperator(
    task_id="inference_op",
    python_callable=run_predictions,
    dag=dag
)

wait_all_files >> choose_model_op >> inference_op