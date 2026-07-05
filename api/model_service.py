from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


WORKSPACE = Path(__file__).resolve().parents[1]
MODEL_PATH = WORKSPACE / "results" / "glu_two_blocks_pooling" / "glu_two_blocks_pooling_model.keras"
WINDOWS_PATH = WORKSPACE / "data" / "processed" / "windows_w10.npz"
TEST_METADATA_PATH = WORKSPACE / "data" / "processed" / "test_window_metadata_w10.csv"

FEATURE_COLUMNS = ["chI", "chV", "chT", "disI", "disV", "BCt", "SOH"]
WINDOW_SIZE = 10
INPUT_DIM = len(FEATURE_COLUMNS)


class GLUBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.projection = tf.keras.layers.Dense(hidden_dim * 2, name="glu_projection")

    def call(self, inputs):
        projected = self.projection(inputs)
        a, b = tf.split(projected, num_or_size_splits=2, axis=-1)
        return a * tf.sigmoid(b)

    def build(self, input_shape):
        self.projection.build(input_shape)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


def model_available() -> bool:
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"GLUBlock": GLUBlock},
        compile=False,
        safe_mode=False,
    )


def validate_window(window: list[list[float]]) -> np.ndarray:
    array = np.asarray(window, dtype=np.float32)
    expected_shape = (WINDOW_SIZE, INPUT_DIM)
    if array.shape != expected_shape:
        raise ValueError(f"Expected window shape {expected_shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Window contains NaN or infinite values.")
    return array


def classify_rul(predicted_rul: float) -> str:
    if predicted_rul <= 20:
        return "critical"
    if predicted_rul <= 50:
        return "warning"
    return "healthy"


def predict_window(window: list[list[float]]) -> dict[str, Any]:
    validated_window = validate_window(window)
    model_input = validated_window[np.newaxis, :, :]
    model = load_model()
    predicted_rul = float(model.predict(model_input, verbose=0).reshape(-1)[0])
    predicted_rul = max(predicted_rul, 0.0)
    return {
        "predicted_rul": predicted_rul,
        "status": classify_rul(predicted_rul),
        "input_shape": list(model_input.shape),
        "feature_columns": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "model_name": "Two GLU Blocks + Pooling",
    }


def load_sample(split: str = "test", index: int = 0) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    if not WINDOWS_PATH.exists():
        raise FileNotFoundError(f"Window data was not found: {WINDOWS_PATH}")

    data = np.load(WINDOWS_PATH, allow_pickle=True)
    x_key = {"train": "X_train", "val": "X_val", "test": "X_test"}[split]
    y_key = {"train": "y_train", "val": "y_val", "test": "y_test"}[split]
    windows = data[x_key].astype(np.float32)
    targets = data[y_key].astype(np.float32)

    if index < 0 or index >= len(windows):
        raise IndexError(f"Sample index {index} is outside the {split} range 0-{len(windows) - 1}.")

    payload: dict[str, Any] = {
        "split": split,
        "index": index,
        "window": windows[index].tolist(),
        "true_rul": float(targets[index]),
        "feature_columns": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
    }

    if split == "test" and TEST_METADATA_PATH.exists():
        metadata = pd.read_csv(TEST_METADATA_PATH)
        if index < len(metadata):
            payload["metadata"] = metadata.iloc[index].to_dict()

    return payload
