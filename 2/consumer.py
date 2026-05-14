import json
import os
import numpy as np
import tritonclient.grpc as grpc_client
from kafka import KafkaConsumer, KafkaProducer

class InferenceWorker:
    def __init__(self):
        self.broker = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.in_topic = os.getenv("INPUT_TOPIC", "task_queue")
        self.out_topic = os.getenv("OUTPUT_TOPIC", "result_queue")
        
        self.consumer = KafkaConsumer(
            self.in_topic,
            bootstrap_servers=[self.broker],
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest"
        )
        self.producer = KafkaProducer(
            bootstrap_servers=[self.broker],
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        
        triton_url = os.getenv("TRITON_URL", "localhost:8001")
        self.triton_stub = grpc_client.InferenceServerClient(url=triton_url)
        self.model_name = "codegen"

    def run_model(self, text_input):
        max_size = 2500
        raw_ids = []
        for char in text_input[:max_size]:
            raw_ids.append(ord(char))
            
        if len(raw_ids) < max_size:
            raw_ids += [0] * (max_size - len(raw_ids))
            
        attn_mask = []
        for val in raw_ids:
            attn_mask.append(1 if val > 0 else 0)

        inputs = [
            grpc_client.InferInput("input_ids", [1, max_size], "INT32"),
            grpc_client.InferInput("attention_mask", [1, max_size], "INT32")
        ]
        inputs[0].set_data_from_numpy(np.array([raw_ids], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.array([attn_mask], dtype=np.int32))

        response = self.triton_stub.infer(self.model_name, inputs, 
                                          outputs=[grpc_client.InferRequestedOutput("output_0")])
        
        output_data = response.as_numpy("output_0")[0]
        decoded_output = []
        for token in output_data:
            if 0 < int(token) < 0x10FFFF:
                decoded_output.append(chr(int(token)))
        
        return "".join(decoded_output).strip()

    def start_processing(self):
        print("--- Воркер запущен. Ожидание данных ---")
        for message in self.consumer:
            try:
                payload = message.value
                prompt = payload.get("text", "")
                
                print(f"[*] Обработка запроса: {prompt[:40]}...")
                
                ai_result = self.run_model(prompt)
                
                response_package = {
                    "request_info": {
                        "prompt_preview": prompt[:100],
                        "length": len(prompt)
                    },
                    "result": {
                        "content": ai_result,
                        "tag": "ai_generated"
                    },
                    "meta": payload.get("meta", {})
                }
                
                self.producer.send(self.out_topic, value=response_package)
                print(f"[DONE] Результат отправлен в {self.out_topic}")
                
            except Exception as e:
                print(f"[!] Ошибка воркера: {e}")

if __name__ == "__main__":
    worker = InferenceWorker()
    worker.start_processing()