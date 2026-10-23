import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from pydantic import BaseModel, Field

SERVICE_NAME = os.getenv("SERVICE_NAME", "backend")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
STARTUP_RETRIES = int(os.getenv("MODEL_LOAD_RETRIES", "10"))

# Must match the training data: same names, same order.
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

log = logging.getLogger("backend")
state: dict = {"model": None, "info": None, "error": None}


def load_champion() -> None:
    """Load the model the alias points at, plus its metadata. Called at startup and by /reload."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    state["model"] = model
    state["info"] = {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "run_name": mv.tags.get("run_name"),
        "metrics": {k: float(v) for k, v in mv.tags.items() if k.startswith("test_")},
        "features": FEATURES,
        "classes": [str(c) for c in model.classes_],
        "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tracking_uri": MLFLOW_TRACKING_URI,
    }
    state["error"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, not on every request. Retry while MLflow / training catch up.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for attempt in range(1, STARTUP_RETRIES + 1):
        try:
            load_champion()
            log.warning("Loaded %s@%s (version %s)", MODEL_NAME, MODEL_ALIAS, state["info"]["version"])
            break
        except Exception as exc:  # MLflow down, or no champion registered yet
            state["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("Model not loaded (attempt %s/%s): %s", attempt, STARTUP_RETRIES, state["error"])
            time.sleep(3)
    yield


app = FastAPI(title="Backend API", lifespan=lifespan)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, le=10, description="Sepal length in cm", examples=[5.1])
    sepal_width: float = Field(gt=0, le=10, description="Sepal width in cm", examples=[3.5])
    petal_length: float = Field(gt=0, le=10, description="Petal length in cm", examples=[1.4])
    petal_width: float = Field(gt=0, le=10, description="Petal width in cm", examples=[0.2])


class Prediction(BaseModel):
    species: str
    probabilities: dict[str, float]
    model_name: str
    model_version: str


@app.get("/health")
def health():
    body = {
        "status": "ok" if state["model"] is not None else "degraded",
        "service": SERVICE_NAME,
        "model_loaded": state["model"] is not None,
        "model_version": state["info"]["version"] if state["info"] else None,
    }
    if state["model"] is None:
        body["error"] = state["error"]
        return JSONResponse(status_code=503, content=body)   # 503: up, but can't serve predictions
    return body


@app.get("/message")
def message():
    return {"message": f"Hello from {SERVICE_NAME}"}


@app.get("/model-info")
def model_info():
    if state["info"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["info"]


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures):
    # Pydantic has already rejected wrong types, missing fields and out-of-range values with a 422.
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if features.petal_width > features.petal_length:
        # Valid shape, but not a plausible flower: a business-rule error, so 400.
        raise HTTPException(status_code=400, detail="petal_width cannot be larger than petal_length")

    row = pd.DataFrame([features.model_dump()], columns=FEATURES)   # same column order as training
    try:
        model = state["model"]
        species = str(model.predict(row)[0])
        probs = model.predict_proba(row)[0]
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error")   # no stack trace to the client

    return Prediction(
        species=species,
        probabilities={str(c): round(float(p), 4) for c, p in zip(model.classes_, probs)},
        model_name=MODEL_NAME,
        model_version=state["info"]["version"],
    )


@app.post("/reload")
def reload():
    """Re-load whichever version the alias points at now, e.g. after promoting a challenger."""
    previous = state["info"]["version"] if state["info"] else None
    try:
        load_champion()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Reload failed: {type(exc).__name__}") from exc
    return {"previous_version": previous, "current_version": state["info"]["version"]}
