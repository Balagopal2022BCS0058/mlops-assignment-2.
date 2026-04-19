from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from churn.config import settings
from churn.features.engineering import get_feature_columns
from churn.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY
from churn.schemas import HealthResponse, PredictionResponse, TicketFeatures
from churn.serving.loader import get_model, get_model_version, load_model
import time


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Churn Prediction API", version="2.0.0", lifespan=lifespan)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

FEATURE_COLS = get_feature_columns()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_name=settings.mlflow_model_name,
        model_version=get_model_version(),
        model_stage=settings.model_stage,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TicketFeatures):
    start = time.time()
    model = get_model()

    row = pd.DataFrame([{col: getattr(features, col, 0) for col in FEATURE_COLS}])
    proba = float(model.predict_proba(row)[0, 1])
    label = proba >= 0.5

    elapsed = time.time() - start
    REQUEST_COUNT.labels(endpoint="predict").inc()
    REQUEST_LATENCY.labels(endpoint="predict").observe(elapsed)

    return PredictionResponse(
        customer_id=features.customer_id,
        churn_probability=round(proba, 4),
        churn_label=label,
        model_version=get_model_version(),
    )


@app.get("/model-info")
def model_info():
    return {
        "model_name": settings.mlflow_model_name,
        "model_version": get_model_version(),
        "model_stage": settings.model_stage,
        "features": FEATURE_COLS,
    }
