import os
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import boto3
import tritonclient.grpc as grpc_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CodegenService")

TRITON_URL = f"{os.getenv('TRITON_HOST', 'triton-server')}:{os.getenv('TRITON_PORT', '8001')}"
MODEL_ID = os.getenv("MODEL_NAME", "codegen")
S3_URL = os.getenv("MINIO_ENDPOINT", "http://minio:9000")

class TextPayload(BaseModel):
    text: str

class InferenceResult(BaseModel):
    summary: str
    metrics: dict

class ServiceState:
    """Контейнер для хранения активных соединений."""
    triton: Optional[grpc_client.InferenceServerClient] = None
    s3: Any = None

state = ServiceState()

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Управление жизненным циклом: подключение к сервисам при старте."""
    logger.info("Initializing cloud dependencies...")
    
    state.s3 = boto3.client(
        "s3",
        endpoint_url=S3_URL,
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "admin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "password123"),
        verify=False,
    )
    
    state.triton = grpc_client.InferenceServerClient(url=TRITON_URL)
    
    max_retries = 15
    for i in range(max_retries):
        try:
            if state.triton.is_model_ready(MODEL_ID):
                logger.info(f"Model '{MODEL_ID}' is active and ready.")
                break
        except Exception:
            logger.warning(f"Waiting for Triton... ({i+1}/{max_retries})")
            time.sleep(3)
    
    yield
    if state.triton:
        state.triton.close()

app = FastAPI(title="ML Inference Service", lifespan=app_lifespan)

def execute_triton_inference(raw_text: str) -> str:
    """Преобразование текста в токены и вызов Triton."""
    tokens = [ord(c) for c in raw_text[:512]]
    tokens += [0] * (512 - len(tokens))
    mask = [1 if t != 0 else 0 for t in tokens]
    
    inputs = [
        grpc_client.InferInput("input_ids", [1, 512], "INT32"),
        grpc_client.InferInput("attention_mask", [1, 512], "INT32")
    ]
    
    inputs[0].set_data_from_numpy(np.array([tokens], dtype=np.int32))
    inputs[1].set_data_from_numpy(np.array([mask], dtype=np.int32))
    
    response = state.triton.infer(MODEL_ID, inputs, outputs=[grpc_client.InferRequestedOutput("output_0")])
    raw_output = response.as_numpy("output_0")
    
    chars = [chr(int(t)) for t in raw_output[0] if 0 < t < 1114111]
    return "".join(chars).strip()

@app.post("/codegen", response_model=InferenceResult)
async def handle_generation(payload: TextPayload):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")
    
    start_time = time.time()
    generated_text = execute_triton_inference(payload.text)
    duration = time.time() - start_time
    
    return InferenceResult(
        summary=generated_text,
        metrics={
            "latency": round(duration, 4),
            "input_chars": len(payload.text),
            "output_chars": len(generated_text)
        }
    )

@app.get("/health")
async def check_health():
    return {"status": "online", "timestamp": time.time()}