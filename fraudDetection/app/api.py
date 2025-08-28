from pathlib import Path
import pandas as pd
import time
from fastapi import FastAPI, HTTPException, Request, Response
from typing import List, Dict, Any
import mlflow, mlflow.sklearn
from pydantic import BaseModel, RootModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST



mlflow.set_tracking_uri("file:./tracking")

ROOT = Path(__file__).resolve().parent.parent
MODEL_URI = str(ROOT / "my_model")  

# Carga modelo
model_loaded = False
model = None
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    model_loaded = True
except Exception as e:

    print("ERROR cargando modelo:", e)

FEATURES = getattr(model, "feature_names_in_", None) if model_loaded else None
THRESHOLD = 0.30 

app = FastAPI(title="Fraud API")

class Record(RootModel[Dict[str, Any]]):
    pass

class Batch(BaseModel):
    inputs: List[Record]

def to_dataframe(batch: Batch) -> pd.DataFrame:
    rows = [r.root for r in batch.inputs]
    df = pd.DataFrame(rows)

    if FEATURES is not None:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")
        df = df.loc[:, FEATURES]
    return df


REQUESTS = Counter(
    "http_requests_total",
    "Total de peticiones HTTP",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latencia de peticiones HTTP por endpoint",
    ["method", "endpoint"]
)

# ----------------------------------

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    method = request.method

    endpoint = request.scope.get("path", "unknown")

    try:
        response: Response = await call_next(request)
        status = str(response.status_code)
    except Exception:

        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        REQUESTS.labels(method=method, endpoint=endpoint, status="500").inc()
        raise

    elapsed = time.perf_counter() - start
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
    REQUESTS.labels(method=method, endpoint=endpoint, status=status).inc()
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/score")
def score(batch: Batch):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    df = to_dataframe(batch)
    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=400, detail="El modelo no soporta predict_proba")
    proba = model.predict_proba(df)[:, 1].tolist()
    return {"probabilities": proba}

@app.post("/predict")
def predict(batch: Batch):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    df = to_dataframe(batch)
    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= THRESHOLD).astype(int).tolist()
    return {"predictions": preds, "probabilities": proba.tolist(), "threshold": THRESHOLD}
