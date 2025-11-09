# app/server.py
import threading
from typing import List, Optional

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------- Config ----------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MODEL_NAME = "iris-classifier"
DEFAULT_MODEL_VERSION = "1"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------- Globals ----------
_model_lock = threading.RLock()
_model_name: str = DEFAULT_MODEL_NAME
_model_version: str = DEFAULT_MODEL_VERSION
_model = None  # lazy-loaded

def _model_uri(name: str, version: str) -> str:
    return f"models:/{name}/{version}"

def _load_model(name: str, version: str):
    uri = _model_uri(name, version)
    return mlflow.pyfunc.load_model(uri), uri

def _ensure_model_loaded():
    global _model
    with _model_lock:
        if _model is None:
            try:
                _model, _ = _load_model(_model_name, _model_version)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Could not load model '{_model_name}' v'{_model_version}' from {MLFLOW_TRACKING_URI}: {e}"
                )
        return _model

# ---------- Schemas ----------
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "samples": [
                    {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},
                    {"sepal_length":6.7,"sepal_width":3.1,"petal_length":4.7,"petal_width":1.5}
                ]
            }]
        }
    }

IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]
    class_label: List[str]
    model_config = {
        "json_schema_extra": {
            "examples": [{"class_id":[0,1], "class_label":["setosa","versicolor"]}]
        }
    }

class ModelVersionResponse(BaseModel):
    model_name: str
    model_version: str
    model_uri: str

class ModelVersionUpdate(BaseModel):
    model_name: Optional[str] = None   # registered name
    model_version: Optional[str] = None  # e.g. "2"

# ---------- App ----------
app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
)

@app.get("/health", tags=["health"])
def health():
    try:
        _ensure_model_loaded()
        return {"status": "ok", "model_uri": _model_uri(_model_name, _model_version)}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.get("/model/version", response_model=ModelVersionResponse, tags=["model"])
def get_model_version():
    return ModelVersionResponse(
        model_name=_model_name,
        model_version=_model_version,
        model_uri=_model_uri(_model_name, _model_version),
    )

@app.put("/model/version", response_model=ModelVersionResponse, tags=["model"])
def update_model_version(update: ModelVersionUpdate):
    global _model, _model_name, _model_version
    new_name = update.model_name or _model_name
    new_version = update.model_version or _model_version
    try:
        loaded_model, _ = _load_model(new_name, new_version)  # validate first
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model {new_name}/{new_version}: {e}")

    with _model_lock:
        _model = loaded_model
        _model_name = new_name
        _model_version = new_version

    return ModelVersionResponse(
        model_name=_model_name,
        model_version=_model_version,
        model_uri=_model_uri(_model_name, _model_version),
    )

@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    if not req.samples:
        raise HTTPException(status_code=400, detail="No samples provided.")
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = pd.DataFrame([s.model_dump() for s in req.samples], columns=cols)

    model = _ensure_model_loaded()
    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    if isinstance(y_pred, (list, tuple)):
        y_pred = np.array(y_pred)
    if getattr(y_pred, "ndim", 1) == 2:
        class_ids = np.argmax(y_pred, axis=1).astype(int).tolist()
    else:
        class_ids = np.asarray(y_pred).astype(int).tolist()

    labels = [IRIS_LABELS.get(i, f"class_{i}") for i in class_ids]
    return PredictResponse(class_id=class_ids, class_label=labels)
