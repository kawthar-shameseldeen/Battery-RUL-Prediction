from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.model_service import (
    FEATURE_COLUMNS,
    INPUT_DIM,
    MODEL_PATH,
    WINDOW_SIZE,
    load_model,
    load_sample,
    model_available,
    predict_window,
)


class PredictionRequest(BaseModel):
    window: list[list[float]] = Field(
        ...,
        description="A scaled battery window with shape (10, 7). Feature order: chI, chV, chT, disI, disV, BCt, SOH.",
    )


class PredictionResponse(BaseModel):
    predicted_rul: float
    status: str
    input_shape: list[int]
    feature_columns: list[str]
    window_size: int
    model_name: str


app = FastAPI(
    title="Battery RUL Prediction API",
    description="FastAPI backend for the selected GLU model: Two GLU Blocks + Pooling.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_load_model() -> None:
    if model_available():
        load_model()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Battery RUL Prediction API",
        "model": "Two GLU Blocks + Pooling",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_available": model_available(),
        "model_path": str(MODEL_PATH),
        "window_size": WINDOW_SIZE,
        "input_dim": INPUT_DIM,
        "feature_columns": FEATURE_COLUMNS,
        "message": (
            "Model is available for live predictions."
            if model_available()
            else "Model file is missing. Restore the .keras file to enable live predictions."
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        return predict_window(request.window)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/sample/{split}/{index}")
def sample(split: str, index: int) -> dict[str, Any]:
    try:
        return load_sample(split=split, index=index)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/predict-sample/{split}/{index}")
def predict_sample(split: str, index: int) -> dict[str, Any]:
    try:
        sample_payload = load_sample(split=split, index=index)
        prediction = predict_window(sample_payload["window"])
        return {
            **prediction,
            "split": split,
            "index": index,
            "true_rul": sample_payload.get("true_rul"),
            "metadata": sample_payload.get("metadata"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
