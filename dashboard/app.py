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
PROCESSED_DATA_PATH = WORKSPACE / "data" / "processed" / "processed_data.csv"
SAMPLE_UPLOAD_PATH = WORKSPACE / "dashboard" / "sample_client_battery_upload.csv"
LOSS_CURVE_PATH = RESULTS_DIR / "loss_curve.png"
TRUE_VS_PRED_PATH = RESULTS_DIR / "true_vs_pred.png"
ERROR_OVER_CYCLES_PATH = RESULTS_DIR / "prediction_error_over_cycles.png"

DEFAULT_API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Battery Health Monitor",
    page_icon="battery",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    .hero-card {
        padding: 1.4rem;
        border-radius: 1.1rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f766e 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 1rem;
    }
    .hero-card h1 {
        margin-bottom: 0.25rem;
    }
    .alert-card {
        padding: 1.25rem;
        border-radius: 1rem;
        border: 1px solid;
        margin-top: 1rem;
    }
    .alert-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .alert-message {
        font-size: 1rem;
        line-height: 1.45;
    }
    .soft-note {
        padding: 0.85rem 1rem;
        border-radius: 0.75rem;
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-top: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics() -> dict[str, Any]:
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_processed_cycle_data() -> pd.DataFrame:
    processed = pd.read_csv(PROCESSED_DATA_PATH)
    feature_columns = ["chI", "chV", "chT", "disI", "disV", "BCt", "SOH"]
    cycle_columns = ["battery_id", "cycle", *feature_columns, "RUL"]
    processed = processed.loc[:, cycle_columns].copy()
    return (
        processed.groupby(["battery_id", "cycle"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["battery_id", "cycle"])
        .reset_index(drop=True)
    )


def prepare_uploaded_cycle_data(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["battery_id", "cycle", "chI", "chV", "chT", "disI", "disV", "BCt", "SOH"]
    missing_columns = [column for column in required_columns if column not in uploaded_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    optional_columns = ["RUL"] if "RUL" in uploaded_df.columns else []
    selected_columns = [*required_columns, *optional_columns]
    upload_data = uploaded_df.loc[:, selected_columns].copy()

    for column in ["cycle", "chI", "chV", "chT", "disI", "disV", "BCt", "SOH", *optional_columns]:
        upload_data[column] = pd.to_numeric(upload_data[column], errors="raise")

    cycle_data = (
        upload_data.groupby(["battery_id", "cycle"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["battery_id", "cycle"])
        .reset_index(drop=True)
    )

    if len(cycle_data) < 10:
        raise ValueError("Uploaded CSV must contain at least 10 battery cycles.")

    return cycle_data


def status_style(status: str) -> dict[str, str]:
    styles = {
        "healthy": {
            "client_label": "Battery is healthy",
            "short_label": "Healthy",
            "color": "#16a34a",
            "background": "rgba(22, 163, 74, 0.13)",
            "message": "The battery still has a good number of cycles remaining. Continue normal use and keep monitoring it.",
            "action": "No immediate action needed.",
        },
        "warning": {
            "client_label": "Battery needs attention",
            "short_label": "Warning",
            "color": "#d97706",
            "background": "rgba(217, 119, 6, 0.14)",
            "message": "The battery is getting closer to its end-of-life range. Plan maintenance, replacement, or closer monitoring.",
            "action": "Prepare a replacement plan.",
        },
        "critical": {
            "client_label": "Battery is critical",
            "short_label": "Critical",
            "color": "#dc2626",
            "background": "rgba(220, 38, 38, 0.14)",
            "message": "The battery is near failure according to the model. It should not be relied on for important operation.",
            "action": "Replace or inspect immediately.",
        },
    }
    return styles.get(status, styles["warning"])


def call_predict_sample(api_url: str, split: str, index: int) -> dict[str, Any]:
    response = requests.get(
        f"{api_url.rstrip('/')}/predict-sample/{split}/{index}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def call_predict_window(api_url: str, window: list[list[float]]) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"window": window},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def render_alert_card(status: str, predicted_rul: float, end_cycle: int) -> None:
    style = status_style(status)
    estimated_failure_cycle = end_cycle + int(round(predicted_rul))

    st.markdown(
        f"""
        <div class="alert-card" style="background: {style['background']}; border-color: {style['color']};">
            <div class="alert-title" style="color: {style['color']};">{style['client_label']}</div>
            <div class="alert-message">{style['message']}</div>
            <div class="soft-note">
                <strong>Recommended action:</strong> {style['action']}<br>
                <strong>Estimated end-of-life cycle:</strong> around cycle {estimated_failure_cycle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(
    *,
    result: dict[str, Any],
    true_rul: float,
    end_cycle: int,
    show_true_values: bool,
) -> None:
    predicted_rul = float(result["predicted_rul"])
    status = result["status"]
    style = status_style(status)

    st.success("Battery prediction completed.")
    card_col1, card_col2, card_col3 = st.columns(3)
    card_col1.metric("Estimated remaining cycles", f"{predicted_rul:.0f}")
    card_col2.metric("Current cycle", f"{end_cycle}")
    card_col3.metric("Alert level", style["short_label"])

    render_alert_card(status, predicted_rul, end_cycle)

    with st.expander("Technical validation values"):
        st.write("These values are useful for project evaluation, but a normal client does not need them.")
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        tech_col1.metric("Predicted RUL", f"{predicted_rul:.2f}")
        if show_true_values:
            tech_col2.metric("True RUL", f"{true_rul:.2f}")
            tech_col3.metric("Prediction error", f"{predicted_rul - true_rul:.2f}")
        else:
            tech_col2.metric("True RUL", "Not available")
            tech_col3.metric("Prediction error", "Not available")


st.markdown(
    """
    <div class="hero-card">
        <h1>Battery Health Monitor</h1>
        <p>Check the battery condition, estimate remaining useful life, and receive a clear maintenance alert.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics = load_metrics()
predictions = load_predictions()
processed_cycle_data = load_processed_cycle_data()
feature_columns = metrics["feature_columns"]
window_size = int(metrics["window_size"])

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("FastAPI URL", DEFAULT_API_URL)
    st.caption("FastAPI must be running before you click predict.")

    st.header("Data")
    data_source = st.radio(
        "Choose source",
        options=[
            "Official test battery",
            "All processed batteries",
            "Upload client CSV",
        ],
    )

    with st.expander("Technical settings"):
        st.write("Window size:", window_size)
        st.write("Features:", ", ".join(feature_columns))
        st.write("Model:", "Two GLU Blocks + Pooling")

top_col1, top_col2, top_col3 = st.columns(3)
top_col1.metric("System status", "Ready")
top_col2.metric("Prediction unit", "Cycles")
top_col3.metric("Alert thresholds", "50 / 20 cycles")

st.markdown("---")

left_col, right_col = st.columns([0.95, 1.05])

with left_col:
    st.subheader("Battery Check")

    if data_source == "Official test battery":
        st.caption("Uses the official unseen test battery: B6.")
        max_index = len(predictions) - 1
        selected_index = st.slider(
            "Select the current battery window",
            min_value=0,
            max_value=max_index,
            value=0,
        )

        selected_row = predictions.iloc[selected_index]
        battery_id = str(selected_row["battery_id"])
        start_cycle = int(selected_row["start_cycle"])
        end_cycle = int(selected_row["end_cycle"])
        true_rul = float(selected_row["true_RUL"])

        st.write(f"Battery **{battery_id}**, cycles **{start_cycle}-{end_cycle}**")

        if st.button("Check Battery Health", type="primary"):
            try:
                result = call_predict_sample(api_url, "test", selected_index)
                render_prediction_result(
                    result=result,
                    true_rul=true_rul,
                    end_cycle=end_cycle,
                    show_true_values=True,
                )
            except requests.exceptions.RequestException as exc:
                st.error("Could not reach FastAPI. Please start the API server first.")
                st.code(str(exc))

    elif data_source == "All processed batteries":
        st.caption("Uses the already-preprocessed project CSV for demonstration across batteries.")
        available_batteries = processed_cycle_data["battery_id"].unique().tolist()
        selected_battery = st.selectbox("Select battery", available_batteries)
        battery_cycle_data = processed_cycle_data[
            processed_cycle_data["battery_id"] == selected_battery
        ].reset_index(drop=True)

        max_window_start = len(battery_cycle_data) - window_size
        selected_start = st.slider(
            "Select the current cycle window",
            min_value=0,
            max_value=max_window_start,
            value=0,
        )

        selected_window = battery_cycle_data.iloc[
            selected_start : selected_start + window_size
        ]
        start_cycle = int(selected_window["cycle"].iloc[0])
        end_cycle = int(selected_window["cycle"].iloc[-1])
        true_rul = float(selected_window["RUL"].iloc[-1])

        st.write(f"Battery **{selected_battery}**, cycles **{start_cycle}-{end_cycle}**")

        if st.button("Check Battery Health", type="primary"):
            try:
                window = selected_window.loc[:, feature_columns].values.tolist()
                result = call_predict_window(api_url, window)
                render_prediction_result(
                    result=result,
                    true_rul=true_rul,
                    end_cycle=end_cycle,
                    show_true_values=True,
                )
            except requests.exceptions.RequestException as exc:
                st.error("Could not reach FastAPI. Please start the API server first.")
                st.code(str(exc))

        with st.expander("Selected model input"):
            st.dataframe(
                selected_window.loc[:, ["battery_id", "cycle", *feature_columns, "RUL"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Upload a preprocessed CSV with the same model feature format.")
        if SAMPLE_UPLOAD_PATH.exists():
            with SAMPLE_UPLOAD_PATH.open("rb") as sample_file:
                st.download_button(
                    "Download sample client CSV",
                    data=sample_file,
                    file_name="sample_client_battery_upload.csv",
                    mime="text/csv",
                )

        uploaded_file = st.file_uploader("Upload battery CSV", type=["csv"])

        if uploaded_file is None:
            st.info("Upload a CSV file to check a client battery.")
            uploaded_cycle_data = pd.DataFrame()
        else:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                uploaded_cycle_data = prepare_uploaded_cycle_data(uploaded_df)
                available_batteries = uploaded_cycle_data["battery_id"].unique().tolist()
                selected_battery = st.selectbox("Select uploaded battery", available_batteries)
                battery_cycle_data = uploaded_cycle_data[
                    uploaded_cycle_data["battery_id"] == selected_battery
                ].reset_index(drop=True)

                max_window_start = len(battery_cycle_data) - window_size
                selected_start = st.slider(
                    "Select the current cycle window",
                    min_value=0,
                    max_value=max_window_start,
                    value=max_window_start,
                )

                selected_window = battery_cycle_data.iloc[
                    selected_start : selected_start + window_size
                ]
                start_cycle = int(selected_window["cycle"].iloc[0])
                end_cycle = int(selected_window["cycle"].iloc[-1])
                true_rul = (
                    float(selected_window["RUL"].iloc[-1])
                    if "RUL" in selected_window.columns
                    else 0.0
                )
                has_true_rul = "RUL" in selected_window.columns

                st.write(f"Battery **{selected_battery}**, cycles **{start_cycle}-{end_cycle}**")

                if st.button("Check Uploaded Battery Health", type="primary"):
                    try:
                        window = selected_window.loc[:, feature_columns].values.tolist()
                        result = call_predict_window(api_url, window)
                        render_prediction_result(
                            result=result,
                            true_rul=true_rul,
                            end_cycle=end_cycle,
                            show_true_values=has_true_rul,
                        )
                    except requests.exceptions.RequestException as exc:
                        st.error("Could not reach FastAPI. Please start the API server first.")
                        st.code(str(exc))

                with st.expander("Uploaded model input"):
                    display_columns = ["battery_id", "cycle", *feature_columns]
                    if has_true_rul:
                        display_columns.append("RUL")
                    st.dataframe(
                        selected_window.loc[:, display_columns],
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as exc:
                st.error("The uploaded CSV could not be used.")
                st.code(str(exc))

with right_col:
    st.subheader("Battery Trend")
    if data_source == "Official test battery":
        trend_data = predictions.set_index("end_cycle")[["true_RUL", "predicted_RUL"]]
        st.line_chart(trend_data)
        st.caption("The trend compares the real RUL and model-predicted RUL for B6.")
    elif data_source == "All processed batteries":
        trend_data = battery_cycle_data.set_index("cycle")[["RUL"]]
        st.line_chart(trend_data)
        st.caption("This chart shows how the true RUL decreases over battery cycles.")
    else:
        if "uploaded_cycle_data" in locals() and not uploaded_cycle_data.empty:
            if "RUL" in battery_cycle_data.columns:
                trend_data = battery_cycle_data.set_index("cycle")[["RUL"]]
                st.line_chart(trend_data)
                st.caption("This chart shows the uploaded battery RUL trend.")
            else:
                feature_trend = battery_cycle_data.set_index("cycle")[["SOH"]]
                st.line_chart(feature_trend)
                st.caption("No true RUL column was uploaded, so the chart shows SOH over cycles.")
        else:
            st.info("Upload a CSV to show the battery trend.")

    st.subheader("Simple Guide")
    st.markdown(
        """
        - **Healthy** means the battery is safe to continue using.
        - **Warning** means the battery should be monitored and replacement should be planned.
        - **Critical** means the battery is near failure and should be inspected or replaced.
        """
    )

with st.expander("Technical model performance"):
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("MAE", f"{metrics['mae']:.2f}")
    metric_col2.metric("RMSE", f"{metrics['rmse']:.2f}")
    metric_col3.metric("R2", f"{metrics['r2']:.3f}")
    metric_col4.metric("Test windows", f"{metrics['test_shape'][0]}")

    st.write(
        "These metrics are kept for developers and project evaluation. "
        "They are hidden here so the client dashboard stays simple."
    )

    if data_source == "Official test battery":
        st.subheader("Prediction Error Over Cycles")
        error_data = predictions.set_index("end_cycle")[["prediction_error"]]
        st.line_chart(error_data)

with st.expander("Experiment figures"):
    fig_col1, fig_col2, fig_col3 = st.columns(3)
    if LOSS_CURVE_PATH.exists():
        fig_col1.image(str(LOSS_CURVE_PATH), caption="Training vs Validation Loss")
    if TRUE_VS_PRED_PATH.exists():
        fig_col2.image(str(TRUE_VS_PRED_PATH), caption="True RUL vs Predicted RUL")
    if ERROR_OVER_CYCLES_PATH.exists():
        fig_col3.image(str(ERROR_OVER_CYCLES_PATH), caption="Prediction Error Over Cycles")

st.caption(
    "Next integration point: after XAI is implemented, this dashboard can show why the model gave each battery alert."
)
