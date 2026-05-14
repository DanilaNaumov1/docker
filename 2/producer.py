import json
import time
import os
from kafka import KafkaProducer

class TaskDispatcher:
    def __init__(self):
        self.broker = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.topic = os.getenv("INPUT_TOPIC", "task_queue")
        
        self.producer = KafkaProducer(
            bootstrap_servers=[self.broker],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3
        )

    def prepare_tasks(self):
        return [
            {"text": "Write a Python script for binary search", "meta": {"task_id": 101}},
            {"text": "Create a CSS flexbox layout for a navigation bar", "meta": {"task_id": 102}},
            {"text": "Implement a basic HTTP server in Node.js", "meta": {"task_id": 103}},
            {"text": "Write a SQL query to join two tables and group by date", "meta": {"task_id": 104}}
        ]

    def start_dispatch(self):
        tasks = self.prepare_tasks()
        print(">>> Запуск диспетчера задач...")
        for i, task in enumerate(tasks):
            try:
                self.producer.send(self.topic, value=task).get(timeout=10)
                print(f"[OK] Задача #{i+1} отправлена в топик {self.topic}")
                time.sleep(1.5)
            except Exception as e:
                print(f"[FAIL] Ошибка при отправке задачи {i+1}: {e}")
        
        self.producer.flush()
        print(">>> Все задачи распределены.")

if __name__ == "__main__":
    dispatcher = TaskDispatcher()
    dispatcher.start_dispatch()