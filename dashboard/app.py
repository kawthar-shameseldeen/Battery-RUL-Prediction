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
XAI_PREDICTIONS_PATH = WORKSPACE / "results" / "xai_glu_two_blocks_pooling" / "explained_samples.csv"
PROCESSED_DATA_PATH = WORKSPACE / "data" / "processed" / "processed_data.csv"
SAMPLE_UPLOAD_PATH = WORKSPACE / "dashboard" / "sample_client_battery_upload.csv"
XAI_DIR = WORKSPACE / "results" / "xai_glu_two_blocks_pooling"
XAI_SUMMARY_PATH = XAI_DIR / "xai_summary.json"
XAI_PERMUTATION_PATH = XAI_DIR / "permutation_importance.csv"
XAI_IG_FEATURE_PATH = XAI_DIR / "integrated_gradients_feature_importance.csv"
XAI_TEMPORAL_PATH = XAI_DIR / "integrated_gradients_temporal_importance.csv"
XAI_PERMUTATION_FIGURE = XAI_DIR / "permutation_importance.png"
XAI_IG_FEATURE_FIGURE = XAI_DIR / "integrated_gradients_feature_importance.png"
XAI_TEMPORAL_FIGURE = XAI_DIR / "integrated_gradients_temporal_importance.png"
XAI_HEATMAP_FIGURE = XAI_DIR / "integrated_gradients_heatmap.png"
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
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 34rem),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.14), transparent 28rem),
            linear-gradient(180deg, #08111f 0%, #0f172a 45%, #111827 100%);
        color: #e5edf6;
    }
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1240px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #0f172a 58%, #0b1120 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #dbeafe;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 16px 45px rgba(2, 6, 23, 0.18);
    }
    [data-testid="stMetricLabel"] {
        color: #9fb2ca;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 800;
    }
    div.stButton > button,
    div.stDownloadButton > button {
        width: 100%;
        border-radius: 0.9rem;
        border: 0;
        color: #06121f;
        background: linear-gradient(135deg, #5eead4 0%, #38bdf8 100%);
        font-weight: 800;
        padding: 0.85rem 1rem;
        box-shadow: 0 16px 35px rgba(56, 189, 248, 0.20);
    }
    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        color: #020617;
        transform: translateY(-1px);
        box-shadow: 0 20px 42px rgba(45, 212, 191, 0.28);
    }
    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stFileUploader"] section,
    [data-testid="stSlider"] {
        border-radius: 0.85rem;
    }
    .sidebar-brand {
        padding: 1rem;
        border-radius: 1.1rem;
        background:
            linear-gradient(135deg, rgba(20, 184, 166, 0.22), rgba(59, 130, 246, 0.12)),
            rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(125, 211, 252, 0.20);
        margin-bottom: 1rem;
    }
    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .sidebar-brand-copy {
        color: #bfdbfe;
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .hero-card {
        padding: 1.8rem;
        border-radius: 1.35rem;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(30, 41, 59, 0.92) 48%, rgba(13, 148, 136, 0.82) 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.14);
        margin-bottom: 1.2rem;
        box-shadow: 0 22px 70px rgba(2, 6, 23, 0.36);
        position: relative;
        overflow: hidden;
    }
    .hero-card:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 999px;
        right: -70px;
        top: -90px;
        background: rgba(94, 234, 212, 0.18);
        filter: blur(2px);
    }
    .hero-card h1 {
        margin-bottom: 0.25rem;
        font-size: 2.3rem;
        letter-spacing: -0.045em;
    }
    .hero-card p {
        max-width: 680px;
        color: #dbeafe;
        font-size: 1.02rem;
    }
    .hero-pill {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(236, 253, 245, 0.12);
        border: 1px solid rgba(167, 243, 208, 0.22);
        color: #a7f3d0;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stat-card {
        padding: 1.05rem;
        border-radius: 1.1rem;
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 18px 44px rgba(2, 6, 23, 0.23);
        min-height: 108px;
    }
    .stat-label {
        color: #93a4ba;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stat-value {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }
    .stat-copy {
        color: #b7c5d8;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    .panel {
        padding: 1.25rem;
        border-radius: 1.2rem;
        background: rgba(15, 23, 42, 0.70);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 18px 46px rgba(2, 6, 23, 0.22);
        margin-bottom: 1rem;
    }
    .panel-title {
        color: #ffffff;
        font-size: 1.24rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .panel-copy {
        color: #9fb2ca;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-bottom: 0.85rem;
    }
    .alert-card {
        padding: 1.3rem;
        border-radius: 1.15rem;
        border: 1px solid;
        margin-top: 1rem;
        box-shadow: 0 18px 42px rgba(2, 6, 23, 0.18);
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
        background: rgba(15, 23, 42, 0.24);
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-top: 0.75rem;
    }
    .guide-card {
        padding: 1.05rem;
        border-radius: 1rem;
        background: rgba(8, 13, 24, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: #dbeafe;
        line-height: 1.6;
    }
    .guide-card strong {
        color: #5eead4;
    }
    .xai-card {
        padding: 1.05rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.18), rgba(15, 23, 42, 0.64));
        border: 1px solid rgba(94, 234, 212, 0.18);
        margin-top: 1rem;
    }
    .xai-title {
        color: #e0f2fe;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .xai-copy {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .xai-chip {
        display: inline-block;
        padding: 0.32rem 0.55rem;
        border-radius: 999px;
        background: rgba(45, 212, 191, 0.12);
        border: 1px solid rgba(45, 212, 191, 0.22);
        color: #99f6e4;
        font-weight: 800;
        margin-right: 0.35rem;
        margin-top: 0.35rem;
    }
    hr {
        border-color: rgba(148, 163, 184, 0.16) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics() -> dict[str, Any]:
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    if XAI_SUMMARY_PATH.exists():
        with XAI_SUMMARY_PATH.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        baseline = summary.get("baseline_metrics", {})
        return {
            "mae": baseline.get("mae", 0.0),
            "rmse": baseline.get("rmse", 0.0),
            "r2": baseline.get("r2", 0.0),
            "feature_columns": summary.get(
                "feature_columns",
                ["chI", "chV", "chT", "disI", "disV", "BCt", "SOH"],
            ),
            "window_size": summary.get("window_size", 10),
            "test_shape": [summary.get("samples_explained", 0), summary.get("window_size", 10), 7],
        }

    st.error(
        "Dashboard metrics were not found. Expected either the GLU metrics file "
        "or the XAI summary file."
    )
    st.stop()


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(PREDICTIONS_PATH)
    elif XAI_PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(XAI_PREDICTIONS_PATH)
    else:
        st.error(
            "Prediction results were not found. Expected either "
            "`results/glu_two_blocks_pooling/test_predictions.csv` or "
            "`results/xai_glu_two_blocks_pooling/explained_samples.csv`."
        )
        st.stop()

    if "prediction_error" not in predictions.columns:
        predictions["prediction_error"] = predictions["predicted_RUL"] - predictions["true_RUL"]
    return predictions


@st.cache_data
def load_processed_cycle_data() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        return pd.DataFrame()

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


@st.cache_data
def load_xai_summary() -> dict[str, Any]:
    if not XAI_SUMMARY_PATH.exists():
        return {}
    with XAI_SUMMARY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_xai_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def classify_prediction_status(predicted_rul: float) -> str:
    if predicted_rul <= 20:
        return "critical"
    if predicted_rul <= 50:
        return "warning"
    return "healthy"


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
    render_xai_client_summary()

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


def render_xai_client_summary() -> None:
    xai_summary = load_xai_summary()
    if not xai_summary:
        return

    top_features = [
        item["feature"]
        for item in xai_summary.get("top_integrated_gradient_features", [])[:3]
    ]
    chips = "".join([f'<span class="xai-chip">{feature}</span>' for feature in top_features])

    st.markdown(
        f"""
        <div class="xai-card">
            <div class="xai-title">Why did the model make this battery alert?</div>
            <div class="xai-copy">
                The explanation module shows that the model mostly uses battery health and capacity-related signals.
                The strongest global contributors are:
                <br>{chips}
                <br><br>
                In simple words, the model is mainly checking how the battery capacity and health condition change across the latest 10-cycle window.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-pill">Predictive Battery Maintenance</div>
        <h1>Battery Health Monitor</h1>
        <p>Check battery condition, estimate remaining useful life, and receive a clear maintenance alert before failure happens.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics = load_metrics()
predictions = load_predictions()
processed_cycle_data = load_processed_cycle_data()
xai_summary = load_xai_summary()
xai_permutation = load_xai_table(XAI_PERMUTATION_PATH)
xai_ig_feature = load_xai_table(XAI_IG_FEATURE_PATH)
xai_temporal = load_xai_table(XAI_TEMPORAL_PATH)
feature_columns = metrics["feature_columns"]
window_size = int(metrics["window_size"])

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Battery RUL Control</div>
            <div class="sidebar-brand-copy">Select a data source, connect to the prediction API, and run a battery health check.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Connection")
    api_url = st.text_input("FastAPI URL", DEFAULT_API_URL)
    st.caption("FastAPI must be running before you click predict.")

    st.subheader("Data Source")
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
with top_col1:
    render_stat_card("System Status", "Ready", "FastAPI connection is configured from the sidebar.")
with top_col2:
    render_stat_card("Prediction Unit", "Cycles", "Remaining useful life is measured in battery cycles.")
with top_col3:
    render_stat_card("Alert Thresholds", "50 / 20", "Warning at 50 cycles, critical at 20 cycles.")

st.markdown("---")

left_col, right_col = st.columns([0.95, 1.05])

with left_col:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Battery Check</div>
            <div class="panel-copy">Choose the battery data and run a health check. The result will be shown as a simple alert.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            predicted_rul = float(selected_row["predicted_RUL"])
            result = {
                "predicted_rul": predicted_rul,
                "status": classify_prediction_status(predicted_rul),
            }
            render_prediction_result(
                result=result,
                true_rul=true_rul,
                end_cycle=end_cycle,
                show_true_values=True,
            )

    elif data_source == "All processed batteries":
        st.caption("Uses the already-preprocessed project CSV for demonstration across batteries.")
        if processed_cycle_data.empty:
            st.warning(
                "`data/processed/processed_data.csv` is missing in this branch, "
                "so this mode is currently unavailable."
            )
            battery_cycle_data = pd.DataFrame()
        else:
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
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Battery Trend</div>
            <div class="panel-copy">Track how the remaining useful life changes as the battery cycles progress.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if data_source == "Official test battery":
        trend_data = predictions.set_index("end_cycle")[["true_RUL", "predicted_RUL"]]
        st.line_chart(trend_data)
        st.caption("The trend compares the real RUL and model-predicted RUL for B6.")
    elif data_source == "All processed batteries":
        if "battery_cycle_data" in locals() and not battery_cycle_data.empty:
            trend_data = battery_cycle_data.set_index("cycle")[["RUL"]]
            st.line_chart(trend_data)
            st.caption("This chart shows how the true RUL decreases over battery cycles.")
        else:
            st.info("Processed battery data is unavailable in this branch.")
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

    st.markdown(
        """
        <div class="guide-card">
            <strong>Healthy</strong> means the battery is safe to continue using.<br>
            <strong>Warning</strong> means the battery should be monitored and replacement should be planned.<br>
            <strong>Critical</strong> means the battery is near failure and should be inspected or replaced.
        </div>
        """,
        unsafe_allow_html=True,
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

with st.expander("Explainable AI: why the model predicts this way"):
    if xai_summary:
        st.write(
            "XAI explains which inputs the selected GLU model relied on most. "
            "This section is useful for supervisors, developers, and technical users."
        )

        xai_col1, xai_col2 = st.columns(2)
        with xai_col1:
            st.subheader("Most important features")
            if not xai_ig_feature.empty:
                chart_data = xai_ig_feature.set_index("feature")[["mean_abs_integrated_gradient"]]
                st.bar_chart(chart_data)
                st.caption(
                    "Integrated gradients show the strongest contribution came from SOH and BCt."
                )
            else:
                st.info("Integrated gradients feature table was not found.")

        with xai_col2:
            st.subheader("Most important time step")
            if not xai_temporal.empty:
                temporal_chart = xai_temporal.set_index("window_step")[["mean_abs_integrated_gradient"]]
                st.line_chart(temporal_chart)
                st.caption(
                    "The last step in the 10-cycle window has the highest attribution, "
                    "which means the model strongly uses the most recent cycle."
                )
            else:
                st.info("Temporal XAI table was not found.")

        st.subheader("Permutation importance")
        if not xai_permutation.empty:
            permutation_display = xai_permutation[
                ["feature", "rmse_increase", "mae_increase", "r2_drop"]
            ].copy()
            st.dataframe(permutation_display, use_container_width=True, hide_index=True)
            st.caption(
                "Permutation importance checks how much performance worsens when each feature is shuffled. "
                "BCt and SOH caused the largest performance drop, so they are the most important globally."
            )
        else:
            st.info("Permutation importance table was not found.")

        st.subheader("Generated XAI figures")
        fig_a, fig_b, fig_c, fig_d = st.columns(4)
        if XAI_PERMUTATION_FIGURE.exists():
            fig_a.image(str(XAI_PERMUTATION_FIGURE), caption="Permutation Importance")
        if XAI_IG_FEATURE_FIGURE.exists():
            fig_b.image(str(XAI_IG_FEATURE_FIGURE), caption="Integrated Gradients Features")
        if XAI_TEMPORAL_FIGURE.exists():
            fig_c.image(str(XAI_TEMPORAL_FIGURE), caption="Temporal Importance")
        if XAI_HEATMAP_FIGURE.exists():
            fig_d.image(str(XAI_HEATMAP_FIGURE), caption="Feature-Time Heatmap")
    else:
        st.info("XAI results were not found. Run `python scripts/explain_glu_two_blocks_pooling.py` first.")

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
