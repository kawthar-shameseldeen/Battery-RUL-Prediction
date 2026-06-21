from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


WORKSPACE = Path(__file__).resolve().parents[1]
RESULTS_DIR = WORKSPACE / "results" / "glu_two_blocks_pooling"
METRICS_PATH = RESULTS_DIR / "metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
LOSS_CURVE_PATH = RESULTS_DIR / "loss_curve.png"
TRUE_VS_PRED_PATH = RESULTS_DIR / "true_vs_pred.png"
ERROR_OVER_CYCLES_PATH = RESULTS_DIR / "prediction_error_over_cycles.png"

DEFAULT_API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Battery RUL Dashboard",
    page_icon="battery",
    layout="wide",
)


@st.cache_data
def load_metrics() -> dict[str, Any]:
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


def status_from_rul(predicted_rul: float) -> str:
    if predicted_rul <= 20:
        return "critical"
    if predicted_rul <= 50:
        return "warning"
    return "healthy"


def status_color(status: str) -> str:
    return {
        "healthy": "#1f9d55",
        "warning": "#c27803",
        "critical": "#c2410c",
    }.get(status, "#334155")


def call_predict_sample(api_url: str, split: str, index: int) -> dict[str, Any]:
    response = requests.get(
        f"{api_url.rstrip('/')}/predict-sample/{split}/{index}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


st.title("Battery Remaining Useful Life Dashboard")
st.caption("Selected model: Two GLU Blocks + Pooling")

metrics = load_metrics()
predictions = load_predictions()

with st.sidebar:
    st.header("API Settings")
    api_url = st.text_input("FastAPI URL", DEFAULT_API_URL)
    st.info("Run FastAPI first, then use this dashboard to request predictions.")

    st.header("Model Input")
    st.write("Window size:", metrics["window_size"])
    st.write("Input features:", ", ".join(metrics["feature_columns"]))

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("MAE", f"{metrics['mae']:.2f}")
summary_col2.metric("RMSE", f"{metrics['rmse']:.2f}")
summary_col3.metric("R2", f"{metrics['r2']:.3f}")
summary_col4.metric("Test Windows", f"{metrics['test_shape'][0]}")

st.markdown("---")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Live Prediction From FastAPI")

    max_index = len(predictions) - 1
    selected_index = st.slider(
        "Choose a test window",
        min_value=0,
        max_value=max_index,
        value=0,
    )

    selected_row = predictions.iloc[selected_index]
    st.write(
        f"Battery **{selected_row['battery_id']}**, cycles "
        f"**{int(selected_row['start_cycle'])}-{int(selected_row['end_cycle'])}**"
    )

    if st.button("Predict Selected Window", type="primary"):
        try:
            result = call_predict_sample(api_url, "test", selected_index)
            predicted_rul = float(result["predicted_rul"])
            status = result["status"]
            color = status_color(status)

            st.success("Prediction received from FastAPI.")
            pred_col, true_col, err_col = st.columns(3)
            pred_col.metric("Predicted RUL", f"{predicted_rul:.2f}")
            true_col.metric("True RUL", f"{float(result['true_rul']):.2f}")
            err_col.metric(
                "Error",
                f"{predicted_rul - float(result['true_rul']):.2f}",
            )

            st.markdown(
                f"""
                <div style="padding: 1rem; border-radius: 0.75rem; background: {color}22; border: 1px solid {color};">
                    <strong>Status:</strong> <span style="color: {color}; text-transform: uppercase;">{status}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except requests.exceptions.RequestException as exc:
            st.error(
                "Could not reach FastAPI. Make sure it is running at the URL in the sidebar."
            )
            st.code(str(exc))

with right_col:
    st.subheader("Saved Test Results")
    display_columns = [
        "battery_id",
        "start_cycle",
        "end_cycle",
        "true_RUL",
        "predicted_RUL",
        "prediction_error",
    ]
    st.dataframe(
        predictions.loc[:, display_columns],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")

st.subheader("Prediction Trend")
trend_data = predictions.set_index("end_cycle")[["true_RUL", "predicted_RUL"]]
st.line_chart(trend_data)

st.subheader("Prediction Error Over Cycles")
error_data = predictions.set_index("end_cycle")[["prediction_error"]]
st.line_chart(error_data)

st.markdown("---")

st.subheader("Generated Model Figures")
fig_col1, fig_col2, fig_col3 = st.columns(3)
if LOSS_CURVE_PATH.exists():
    fig_col1.image(str(LOSS_CURVE_PATH), caption="Training vs Validation Loss")
if TRUE_VS_PRED_PATH.exists():
    fig_col2.image(str(TRUE_VS_PRED_PATH), caption="True RUL vs Predicted RUL")
if ERROR_OVER_CYCLES_PATH.exists():
    fig_col3.image(str(ERROR_OVER_CYCLES_PATH), caption="Prediction Error Over Cycles")

st.markdown("---")
st.caption(
    "Next integration point: after XAI is implemented, this dashboard can call an /explain endpoint and display feature importance."
)
