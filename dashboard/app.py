from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

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
CANDIDATE_TRAINING_SCRIPT = WORKSPACE / "scripts" / "train_candidate_glu.py"
MODEL_REGISTRY_DIR = WORKSPACE / "model_registry"
CANDIDATE_UPLOAD_DIR = MODEL_REGISTRY_DIR / "uploads"
CANDIDATE_RESULTS_DIR = MODEL_REGISTRY_DIR / "candidates"

DEFAULT_API_URL = "http://127.0.0.1:8000"
PAGE_SLUGS = {
    "overview": "Overview",
    "prediction": "Prediction",
    "training": "Training",
    "insights": "Insights",
}
NAV_ITEMS = [
    ("overview", "D", "Dashboard"),
    ("prediction", "P", "Prediction"),
    ("training", "T", "Training"),
    ("insights", "I", "Insights"),
]


st.set_page_config(
    page_title="Battery Health Monitor",
    page_icon="battery",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');

    :root {
        --ink: #1d1a17;
        --muted: #756b62;
        --paper: rgba(255, 255, 255, 0.88);
        --paper-soft: rgba(250, 246, 240, 0.82);
        --accent: #ea744a;
        --dark: #4f4f4f;
        --dark-2: #3f3f3f;
    }

    html, body, [class*="css"] {
        font-family: "Manrope", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.86), transparent 30rem),
            radial-gradient(circle at top right, rgba(234, 116, 74, 0.13), transparent 20rem),
            linear-gradient(180deg, #d9d3cc 0%, #cec8c0 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"],
    [data-testid="stSidebar"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .main .block-container {
        max-width: 1320px;
        padding-top: 0.45rem;
        padding-bottom: 2rem;
    }

    .app-layout {
        display: grid;
        grid-template-columns: 112px minmax(0, 1fr);
        gap: 1.35rem;
        align-items: start;
    }

    .rail-shell {
        background: linear-gradient(180deg, #5c5c5c 0%, #4d4d4d 28%, #444444 100%);
        border-radius: 2.2rem;
        width: 100%;
        max-width: 102px;
        margin: 0 auto;
        min-height: 84vh;
        padding: 1rem 0.75rem 1.1rem;
        box-shadow: 0 28px 64px rgba(20, 20, 20, 0.24);
        position: fixed;
        left: max(1rem, calc((100vw - 1320px) / 2 + 0.5rem));
        top: 1rem;
        z-index: 20;
    }

    .rail-top {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }

    .rail-brand {
        width: 3.65rem;
        height: 3.65rem;
        border-radius: 999px;
        background: linear-gradient(180deg, #ffffff 0%, #f4ede5 100%);
        color: var(--accent);
        display: grid;
        place-items: center;
        font-size: 1.35rem;
        font-weight: 800;
        box-shadow: 0 10px 22px rgba(18, 18, 18, 0.16);
    }

    .rail-nav {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.9rem;
    }

    .rail-link,
    .rail-link:visited,
    .rail-link:hover,
    .rail-link:active {
        text-decoration: none !important;
    }

    .rail-link {
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.36rem;
        color: rgba(255,255,255,0.84);
        padding: 0.32rem 0.1rem;
        border-radius: 1.3rem;
        transition: 0.18s ease;
    }

    .rail-link:hover {
        background: rgba(255,255,255,0.05);
    }

    .rail-icon {
        width: 2.7rem;
        height: 2.7rem;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: rgba(255,255,255,0.15);
        color: white;
        font-size: 0.95rem;
        font-weight: 800;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
    }

    .rail-link.active .rail-icon {
        background: linear-gradient(180deg, #ffffff 0%, #f4ede5 100%);
        color: var(--accent);
        box-shadow: 0 12px 22px rgba(20, 20, 20, 0.15);
    }

    .rail-label {
        color: rgba(255,255,255,0.78);
        font-size: 0.72rem;
        font-weight: 700;
        line-height: 1.15;
        text-align: center;
    }

    .rail-link.active .rail-label {
        color: white;
    }

    .content-shell {
        min-width: 0;
    }

    .dashboard-shell {
        background: linear-gradient(180deg, rgba(255,255,255,0.56) 0%, rgba(255,255,255,0.42) 100%);
        border: 1px solid rgba(255,255,255,0.46);
        border-radius: 2rem;
        padding: 1.25rem;
        box-shadow: 0 28px 76px rgba(82, 70, 60, 0.12);
        backdrop-filter: blur(20px);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.28fr 0.82fr;
        gap: 1rem;
    }

    .hero-panel,
    .chat-panel,
    .ui-card,
    .dark-card {
        border-radius: 1.65rem;
        overflow: hidden;
    }

    .hero-panel,
    .ui-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(247,243,237,0.84));
        border: 1px solid rgba(255,255,255,0.65);
        padding: 1.4rem;
    }

    .chat-panel {
        background:
            radial-gradient(circle at top center, rgba(234,116,74,0.2), transparent 36%),
            linear-gradient(180deg, rgba(248,245,241,0.9), rgba(240,234,226,0.88));
        border: 1px solid rgba(255,255,255,0.65);
        padding: 1.35rem;
    }

    .hero-kicker,
    .chat-chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.4rem 0.76rem;
        font-size: 0.74rem;
        font-weight: 800;
        background: rgba(234,116,74,0.14);
        color: var(--accent);
        margin-bottom: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .chat-chip {
        background: linear-gradient(135deg, #f2946c, #e36b3e);
        color: white;
        text-transform: none;
        letter-spacing: 0;
    }

    .hero-title {
        font-size: clamp(2rem, 3vw, 3.2rem);
        line-height: 1.02;
        letter-spacing: -0.05em;
        margin: 0 0 0.7rem 0;
        font-weight: 700;
        color: var(--ink);
        max-width: 12ch;
    }

    .hero-copy,
    .card-copy {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.62;
    }

    .progress-banner {
        margin-top: 1.2rem;
        padding: 0.85rem 1rem;
        border-radius: 1.3rem;
        background: rgba(247,242,236,0.92);
        border: 1px solid rgba(58, 48, 42, 0.06);
    }

    .progress-meta {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.65rem;
        font-weight: 700;
    }

    .track {
        height: 0.85rem;
        border-radius: 999px;
        background: rgba(70, 60, 55, 0.09);
        overflow: hidden;
    }

    .fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #f1a07f 0%, #ea744a 100%);
    }

    .bubble {
        background: rgba(255,255,255,0.9);
        border-radius: 1.1rem;
        padding: 0.86rem 0.92rem;
        box-shadow: 0 14px 28px rgba(80, 68, 58, 0.08);
        margin-bottom: 0.7rem;
        color: #332d28;
    }

    .bubble-label,
    .card-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.28rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .card-title {
        color: var(--ink);
        font-size: 1.18rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        letter-spacing: -0.03em;
    }

    .dark-card {
        background: linear-gradient(180deg, #4d4c4c 0%, #393939 100%);
        border: 1px solid rgba(255,255,255,0.15);
        padding: 1.15rem;
        color: white;
    }

    .dark-card .card-label,
    .dark-card .card-title,
    .dark-card .card-copy {
        color: white;
    }

    .footer-note {
        margin-top: 1rem;
        border-radius: 1.25rem;
        padding: 0.9rem 1rem;
        background: rgba(57, 50, 45, 0.92);
        color: rgba(255,255,255,0.9);
        font-size: 0.86rem;
        line-height: 1.55;
    }

    .section-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(245,240,234,0.78));
        border-radius: 1.5rem;
        padding: 1.1rem;
        border: 1px solid rgba(255,255,255,0.64);
        margin-bottom: 1rem;
        box-shadow: 0 14px 34px rgba(83, 71, 62, 0.08);
    }

    .section-title {
        color: var(--ink);
        font-size: 1.24rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.25rem;
    }

    .section-copy {
        color: var(--muted);
        line-height: 1.55;
        margin-bottom: 0.95rem;
    }

    .status-banner {
        border-radius: 1.35rem;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
    }

    .status-title {
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .status-copy {
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .ring-wrap {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
    }

    .ring-card {
        background: rgba(255,255,255,0.72);
        border-radius: 1.35rem;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.64);
        text-align: center;
    }

    .ring-visual {
        width: 9.5rem;
        height: 9.5rem;
        border-radius: 999px;
        margin: 0.65rem auto 0.85rem;
        display: grid;
        place-items: center;
        position: relative;
        background: conic-gradient(var(--ring-color) calc(var(--ring-percent) * 1%), rgba(87, 72, 62, 0.10) 0);
    }

    .ring-visual::before {
        content: "";
        position: absolute;
        inset: 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.92);
        box-shadow: inset 0 0 0 1px rgba(83, 71, 62, 0.06);
    }

    .ring-center {
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        line-height: 1;
    }

    .ring-percent {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .ring-value {
        color: var(--ink);
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.05em;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        width: 100%;
        border: 0;
        border-radius: 1rem;
        background: linear-gradient(135deg, #ef9a79 0%, #e36a3d 100%);
        color: white;
        font-weight: 800;
        box-shadow: 0 14px 28px rgba(227, 106, 61, 0.2);
        padding: 0.82rem 1rem;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.68);
        border: 1px solid rgba(255,255,255,0.54);
        border-radius: 1.2rem;
        padding: 0.85rem 1rem;
        box-shadow: 0 12px 30px rgba(83, 71, 62, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 800;
    }

    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stFileUploader"] section,
    [data-testid="stNumberInput"] input {
        border-radius: 1rem !important;
        background: rgba(255,255,255,0.74) !important;
        border: 1px solid rgba(83, 71, 62, 0.12) !important;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #6a5c50 !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] div {
        color: #6a5c50 !important;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(180deg, rgba(255, 243, 247, 0.98), rgba(255, 236, 242, 0.95)) !important;
        color: #8d6170 !important;
        border: 1px solid rgba(222, 170, 186, 0.42) !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: rgba(255, 232, 239, 0.98) !important;
        color: #744f5b !important;
    }

    [data-baseweb="select"] input,
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #2d2722 !important;
    }

    [data-baseweb="slider"] * {
        color: #6a5c50 !important;
    }

    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] span,
    [data-testid="stCheckbox"] div {
        color: #6a5c50 !important;
    }

    [data-testid="stCheckbox"] input {
        accent-color: #e9a6b7 !important;
    }

    [data-testid="stCheckbox"] [role="checkbox"] {
        background: linear-gradient(180deg, rgba(255, 243, 247, 0.98), rgba(255, 236, 242, 0.95)) !important;
        border: 1px solid rgba(222, 170, 186, 0.42) !important;
        box-shadow: none !important;
    }

    [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
        background: #e9a6b7 !important;
        border-color: #e9a6b7 !important;
    }

    [data-testid="stExpander"] details summary,
    [data-testid="stExpander"] details summary *,
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
        color: #6a5c50 !important;
        fill: #6a5c50 !important;
    }

    [data-testid="stNumberInput"] > div {
        background: linear-gradient(180deg, rgba(255, 243, 247, 0.98), rgba(255, 236, 242, 0.95)) !important;
        border: 1px solid rgba(222, 170, 186, 0.42) !important;
        border-radius: 1rem !important;
        overflow: hidden !important;
        box-shadow: 0 10px 24px rgba(212, 169, 182, 0.12) !important;
    }

    [data-testid="stNumberInput"] [data-baseweb="input"] {
        background: transparent !important;
    }

    [data-testid="stNumberInput"] [data-baseweb="base-input"] {
        background: transparent !important;
    }

    [data-testid="stNumberInput"] [data-baseweb="base-input"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] div[data-baseweb] {
        background: transparent !important;
    }

    [data-testid="stNumberInput"] input {
        color: #6d4d58 !important;
        background: transparent !important;
        box-shadow: none !important;
        font-weight: 700 !important;
    }

    [data-testid="stNumberInput"] button {
        background: rgba(255, 248, 250, 0.98) !important;
        color: #9a6677 !important;
        border-left: 1px solid rgba(222, 170, 186, 0.35) !important;
        border-radius: 0 !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background: rgba(255, 236, 242, 1) !important;
        color: #7f5563 !important;
    }

    [data-testid="stNumberInput"] svg {
        fill: currentColor !important;
    }

    @media (max-width: 1100px) {
        .app-layout {
            grid-template-columns: 1fr;
        }

        .rail-shell {
            position: static;
            min-height: auto;
            max-width: none;
            width: 100%;
        }

        .rail-nav {
            flex-direction: row;
            justify-content: center;
            flex-wrap: wrap;
        }

        .rail-link {
            width: auto;
            min-width: 5.2rem;
        }

        .hero-grid,
        .ring-wrap {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics() -> dict[str, Any]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    if XAI_SUMMARY_PATH.exists():
        summary = json.loads(XAI_SUMMARY_PATH.read_text(encoding="utf-8"))
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

    st.error("Dashboard metrics were not found.")
    st.stop()


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(PREDICTIONS_PATH)
    elif XAI_PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(XAI_PREDICTIONS_PATH)
    else:
        st.error("Prediction results were not found.")
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
    return json.loads(XAI_SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_xai_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_candidate_summaries() -> list[dict[str, Any]]:
    if not CANDIDATE_RESULTS_DIR.exists():
        return []

    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(
        CANDIDATE_RESULTS_DIR.glob("*/comparison_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["candidate_dir"] = str(summary_path.parent)
        payload["candidate_name"] = summary_path.parent.name
        payload["modified_time"] = datetime.fromtimestamp(summary_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        summaries.append(payload)
    return summaries


def safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename.strip())
    return cleaned or "uploaded_dataset.csv"


def save_training_upload(uploaded_file) -> Path:
    CANDIDATE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_path = CANDIDATE_UPLOAD_DIR / f"{timestamp}_{safe_filename(uploaded_file.name)}"
    upload_path.write_bytes(uploaded_file.getbuffer())
    return upload_path


def run_candidate_training(dataset_path: Path, epochs: int, patience: int, cross_validate: bool) -> subprocess.CompletedProcess:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = CANDIDATE_RESULTS_DIR / f"candidate_{timestamp}"
    command = [
        sys.executable,
        str(CANDIDATE_TRAINING_SCRIPT),
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
    ]
    if cross_validate:
        command.append("--cross-validate")
    return subprocess.run(
        command,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=3600,
    )


def prepare_uploaded_cycle_data(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["battery_id", "cycle", "chI", "chV", "chT", "disI", "disV", "BCt", "SOH"]
    missing_columns = [column for column in required_columns if column not in uploaded_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    optional_columns = ["RUL"] if "RUL" in uploaded_df.columns else []
    upload_data = uploaded_df.loc[:, [*required_columns, *optional_columns]].copy()

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
            "color": "#3a8d5c",
            "background": "rgba(58, 141, 92, 0.12)",
            "message": "The battery still has a comfortable cycle reserve and can continue normal operation.",
            "action": "No immediate action needed.",
        },
        "warning": {
            "client_label": "Battery needs attention",
            "short_label": "Warning",
            "color": "#c9862b",
            "background": "rgba(201, 134, 43, 0.12)",
            "message": "The remaining useful life is narrowing, so this battery should be monitored more closely.",
            "action": "Prepare a replacement or maintenance plan.",
        },
        "critical": {
            "client_label": "Battery is critical",
            "short_label": "Critical",
            "color": "#cb5841",
            "background": "rgba(203, 88, 65, 0.12)",
            "message": "The battery is close to failure according to the current model and should not be trusted for critical work.",
            "action": "Inspect or replace immediately.",
        },
    }
    return styles.get(status, styles["warning"])


def classify_prediction_status(predicted_rul: float) -> str:
    if predicted_rul <= 20:
        return "critical"
    if predicted_rul <= 50:
        return "warning"
    return "healthy"


def call_predict_window(api_url: str, window: list[list[float]]) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"window": window},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def ring_card(label: str, value: float, total: float, accent: str, caption: str) -> str:
    total = max(total, 1.0)
    normalized = min(max(value / total, 0.0), 1.0)
    percent = int(round(normalized * 100))
    return textwrap.dedent(
        f"""
        <div class="ring-card">
            <div class="card-label">{label}</div>
            <div class="ring-visual" style="--ring-percent:{percent}; --ring-color:{accent};">
                <div class="ring-center">
                    <div class="ring-percent">{percent}%</div>
                    <div class="ring-value">{int(round(value))}</div>
                </div>
            </div>
            <div class="card-copy">{caption}</div>
        </div>
        """
    ).strip()


def render_section_header(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(status: str, predicted_rul: float, end_cycle: int) -> None:
    style = status_style(status)
    estimated_failure_cycle = end_cycle + int(round(predicted_rul))
    st.markdown(
        f"""
        <div class="status-banner" style="background:{style['background']}; border-color:{style['color']};">
            <div class="status-title" style="color:{style['color']};">{style['client_label']}</div>
            <div class="status-copy">
                {style['message']} Estimated end-of-life cycle is around <strong>{estimated_failure_cycle}</strong>.
                Recommended action: <strong>{style['action']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_xai_client_summary() -> None:
    xai_summary = load_xai_summary()
    if not xai_summary:
        return
    top_features = [item["feature"] for item in xai_summary.get("top_integrated_gradient_features", [])[:3]]
    chips = ", ".join(top_features) if top_features else "SOH, BCt, disV"
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">Why The Model Reached This Alert</div>
            <div class="section-copy">
                The strongest global drivers for this model are battery health and capacity-related signals across the latest 10-cycle window. Most important features: <strong>{chips}</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_technical_summary_card(result_payload: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Technical Validation Values</div>
            <div class="section-copy">Compact validation values for internal review.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    tech_col1.metric("Predicted RUL", f"{float(result_payload['predicted_rul']):.2f}")
    if result_payload.get("show_true_values", False):
        true_rul = float(result_payload.get("true_rul", 0.0))
        predicted_rul = float(result_payload["predicted_rul"])
        tech_col2.metric("True RUL", f"{true_rul:.2f}")
        tech_col3.metric("Prediction error", f"{predicted_rul - true_rul:.2f}")
    else:
        tech_col2.metric("True RUL", "Not available")
        tech_col3.metric("Prediction error", "Not available")


def render_result_panel(result_payload: dict[str, Any]) -> None:
    render_result_panel_with_options(
        result_payload=result_payload,
        show_xai=True,
        show_technical=True,
    )


def render_result_panel_with_options(
    *,
    result_payload: dict[str, Any],
    show_xai: bool,
    show_technical: bool,
) -> None:
    predicted_rul = float(result_payload["predicted_rul"])
    true_rul = float(result_payload.get("true_rul", 0.0))
    end_cycle = int(result_payload["end_cycle"])
    status = result_payload["status"]
    show_true_values = bool(result_payload.get("show_true_values", False))
    total_life = max(end_cycle + predicted_rul, 1.0)
    spent_cycles = min(end_cycle, total_life)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Estimated Remaining Cycles", f"{predicted_rul:.0f}")
    metric_col2.metric("Current Cycle", f"{end_cycle}")
    metric_col3.metric("Alert Level", status_style(status)["short_label"])

    render_status_banner(status, predicted_rul, end_cycle)
    st.markdown(
        "<div class='ring-wrap'>"
        + ring_card("Remaining RUL", predicted_rul, total_life, "#eb7a52", "Cycles still available before failure.")
        + ring_card("Spent RUL", spent_cycles, total_life, "#57514d", "Cycles already consumed from the estimated life.")
        + "</div>",
        unsafe_allow_html=True,
    )
    if show_xai:
        render_xai_client_summary()

    if show_technical:
        with st.expander("Technical validation values"):
            tech_col1, tech_col2, tech_col3 = st.columns(3)
            tech_col1.metric("Predicted RUL", f"{predicted_rul:.2f}")
            if show_true_values:
                tech_col2.metric("True RUL", f"{true_rul:.2f}")
                tech_col3.metric("Prediction error", f"{predicted_rul - true_rul:.2f}")
            else:
                tech_col2.metric("True RUL", "Not available")
                tech_col3.metric("Prediction error", "Not available")


def compute_overview_snapshot(predictions: pd.DataFrame, processed_cycle_data: pd.DataFrame) -> dict[str, Any]:
    spotlight_row = predictions.iloc[len(predictions) // 2]
    battery_count = (
        int(processed_cycle_data["battery_id"].nunique())
        if not processed_cycle_data.empty
        else int(predictions["battery_id"].nunique())
    )
    total_life = float(spotlight_row["end_cycle"] + spotlight_row["predicted_RUL"])
    return {
        "spotlight_predicted_rul": float(spotlight_row["predicted_RUL"]),
        "average_predicted_rul": float(predictions["predicted_RUL"].mean()),
        "average_error": float(predictions["prediction_error"].abs().mean()),
        "battery_count": battery_count,
        "window_count": int(len(predictions)),
        "total_life": total_life,
    }


def get_current_page() -> str:
    page_slug = str(st.query_params.get("page", "overview")).strip().lower()
    return PAGE_SLUGS.get(page_slug, "Overview")


def render_visual_rail(current_page: str) -> None:
    nav_links = []
    for slug, icon, label in NAV_ITEMS:
        is_active = PAGE_SLUGS[slug] == current_page
        active_class = " active" if is_active else ""
        nav_links.append(
            f'<a class="rail-link{active_class}" href="?page={quote_plus(slug)}">'
            f'<span class="rail-icon">{icon}</span>'
            f'<span class="rail-label">{label}</span>'
            "</a>"
        )

    st.markdown(
        f"""
        <div class="rail-shell">
            <div class="rail-top">
                <div class="rail-brand">B</div>
            </div>
            <div class="rail-nav">
                {"".join(nav_links)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_page(metrics: dict[str, Any], predictions: pd.DataFrame, processed_cycle_data: pd.DataFrame) -> None:
    snapshot = compute_overview_snapshot(predictions, processed_cycle_data)
    remaining = snapshot["spotlight_predicted_rul"]
    total_life = snapshot["total_life"]
    progress_pct = min(max(remaining / max(total_life, 1) * 100, 0), 100)

    st.markdown(
        f"""
        <div class="hero-grid">
            <div class="hero-panel">
                <div class="hero-kicker">Problem And Solution</div>
                <div class="hero-title">Predict battery failure before it becomes costly.</div>
                <div class="hero-copy">
                    Battery degradation is hard to estimate from raw cycle readings alone. This project uses a GLU-based Remaining Useful Life model to turn battery behavior into an earlier maintenance signal.
                </div>
                <div class="progress-banner">
                    <div class="progress-meta">
                        <span>Example remaining battery life</span>
                        <span>{remaining:.0f} / {total_life:.0f} cycles</span>
                    </div>
                    <div class="track"><div class="fill" style="width:{progress_pct:.1f}%"></div></div>
                </div>
            </div>
            <div class="chat-panel">
                <div class="chat-chip">Project Overview</div>
                <div class="bubble">
                    <div class="bubble-label">Problem</div>
                    Battery systems can fail unexpectedly when degradation is not tracked precisely over time.
                </div>
                <div class="bubble">
                    <div class="bubble-label">Solution</div>
                    The approved GLU model turns battery cycle data into a predicted remaining useful life estimate.
                </div>
                <div class="bubble">
                    <div class="bubble-label">Outcome</div>
                    Users can detect risk earlier, schedule maintenance more intelligently, and act before unexpected failure happens.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="footer-note">
            The approved model currently reports an MAE of <strong>{metrics['mae']:.1f}</strong>. Continue to <strong>Prediction</strong> to test battery windows, <strong>Training</strong> to retrain candidates, and <strong>Insights</strong> for technical evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_prediction_context(
    data_source: str,
    api_url: str,
    predictions: pd.DataFrame,
    processed_cycle_data: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    battery_cycle_data: pd.DataFrame | None = None
    uploaded_cycle_data: pd.DataFrame | None = None

    if data_source == "Official test battery":
        max_index = len(predictions) - 1
        selected_index = st.slider(
            "Select the current official battery window",
            min_value=0,
            max_value=max_index,
            value=len(predictions) // 2,
        )
        selected_row = predictions.iloc[selected_index]
        st.caption(
            f"Battery {selected_row['battery_id']} • cycles {int(selected_row['start_cycle'])}-{int(selected_row['end_cycle'])}"
        )
        if st.button("Run Health Check", type="primary", key="official_health_check"):
            predicted_rul = float(selected_row["predicted_RUL"])
            st.session_state["latest_prediction_result"] = {
                "predicted_rul": predicted_rul,
                "true_rul": float(selected_row["true_RUL"]),
                "end_cycle": int(selected_row["end_cycle"]),
                "status": classify_prediction_status(predicted_rul),
                "show_true_values": True,
            }
        return battery_cycle_data, uploaded_cycle_data

    if data_source == "All processed batteries":
        if processed_cycle_data.empty:
            st.warning("`data/processed/processed_data.csv` is missing, so this mode is unavailable.")
            return None, None

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
            value=max_window_start,
        )
        selected_window = battery_cycle_data.iloc[selected_start : selected_start + window_size]
        end_cycle = int(selected_window["cycle"].iloc[-1])
        true_rul = float(selected_window["RUL"].iloc[-1])

        st.caption(f"Battery {selected_battery} • cycles {int(selected_window['cycle'].iloc[0])}-{end_cycle}")
        if st.button("Run Health Check", type="primary", key="processed_health_check"):
            try:
                result = call_predict_window(api_url, selected_window.loc[:, feature_columns].values.tolist())
                st.session_state["latest_prediction_result"] = {
                    "predicted_rul": float(result["predicted_rul"]),
                    "true_rul": true_rul,
                    "end_cycle": end_cycle,
                    "status": result["status"],
                    "show_true_values": True,
                }
            except requests.exceptions.RequestException as exc:
                st.error("Could not reach FastAPI. Please start the API server first.")
                st.code(str(exc))

        with st.expander("Selected model input window"):
            st.dataframe(
                selected_window.loc[:, ["battery_id", "cycle", *feature_columns, "RUL"]],
                use_container_width=True,
                hide_index=True,
            )
        return battery_cycle_data, uploaded_cycle_data

    st.caption("Upload a preprocessed CSV with the same feature format as the model.")
    if SAMPLE_UPLOAD_PATH.exists():
        with SAMPLE_UPLOAD_PATH.open("rb") as sample_file:
            st.download_button(
                "Download sample client CSV",
                data=sample_file,
                file_name="sample_client_battery_upload.csv",
                mime="text/csv",
            )

    uploaded_file = st.file_uploader("Upload battery CSV", type=["csv"], key="client_csv_upload")
    if uploaded_file is None:
        st.info("Upload a CSV file to check a client battery.")
        return None, None

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
        selected_window = battery_cycle_data.iloc[selected_start : selected_start + window_size]
        end_cycle = int(selected_window["cycle"].iloc[-1])
        has_true_rul = "RUL" in selected_window.columns
        true_rul = float(selected_window["RUL"].iloc[-1]) if has_true_rul else 0.0

        st.caption(f"Battery {selected_battery} • cycles {int(selected_window['cycle'].iloc[0])}-{end_cycle}")
        if st.button("Run Health Check", type="primary", key="upload_health_check"):
            try:
                result = call_predict_window(api_url, selected_window.loc[:, feature_columns].values.tolist())
                st.session_state["latest_prediction_result"] = {
                    "predicted_rul": float(result["predicted_rul"]),
                    "true_rul": true_rul,
                    "end_cycle": end_cycle,
                    "status": result["status"],
                    "show_true_values": has_true_rul,
                }
            except requests.exceptions.RequestException as exc:
                st.error("Could not reach FastAPI. Please start the API server first.")
                st.code(str(exc))

        with st.expander("Uploaded model input window"):
            display_columns = ["battery_id", "cycle", *feature_columns]
            if has_true_rul:
                display_columns.append("RUL")
            st.dataframe(selected_window.loc[:, display_columns], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("The uploaded CSV could not be used.")
        st.code(str(exc))

    return battery_cycle_data, uploaded_cycle_data


def render_prediction_trend(data_source: str, predictions: pd.DataFrame, battery_cycle_data: pd.DataFrame | None) -> None:
    if data_source == "Official test battery":
        st.line_chart(predictions.set_index("end_cycle")[["true_RUL", "predicted_RUL"]])
        st.caption("Official unseen test battery B6: real RUL against model prediction.")
        return

    if battery_cycle_data is None or battery_cycle_data.empty:
        st.info("Select a battery window to view its trend.")
        return

    if "RUL" in battery_cycle_data.columns:
        st.line_chart(battery_cycle_data.set_index("cycle")[["RUL"]])
        st.caption("This view shows how RUL changes across the selected battery cycles.")
    else:
        st.line_chart(battery_cycle_data.set_index("cycle")[["SOH"]])
        st.caption("No true RUL column was provided, so SOH is shown instead.")


def render_prediction_page(api_url: str, metrics: dict[str, Any], predictions: pd.DataFrame, processed_cycle_data: pd.DataFrame) -> None:
    feature_columns = metrics["feature_columns"]
    window_size = int(metrics["window_size"])

    render_section_header(
        "Prediction",
        "Use this page to test battery windows and turn cycle data into a battery health estimate.",
    )

    result_payload = st.session_state.get("latest_prediction_result")
    control_col, detail_col = st.columns([0.98, 1.02], gap="large")
    with control_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Battery Window Setup</div>
                <div class="section-copy">Choose a source, select the current battery window, and run the health check.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        source = st.selectbox(
            "Prediction source",
            ["Official test battery", "All processed batteries", "Upload client CSV"],
        )
        battery_cycle_data, _ = get_prediction_context(
            source,
            api_url,
            predictions,
            processed_cycle_data,
            feature_columns,
            window_size,
        )
        result_payload = st.session_state.get("latest_prediction_result")

    with detail_col:
        if result_payload:
            predicted_rul = float(result_payload["predicted_rul"])
            style = status_style(result_payload["status"])
            end_cycle = int(result_payload["end_cycle"])
            estimated_failure_cycle = end_cycle + int(round(predicted_rul))
            st.markdown(
                f"""
                <div class="chat-panel">
                    <div class="chat-chip">Health Check Result</div>
                    <div class="card-title" style="font-size:1.9rem; margin-bottom:0.6rem;">
                        {style['client_label']} with about {predicted_rul:.0f} cycles remaining.
                    </div>
                    <div class="card-copy" style="margin-bottom:1rem;">
                        {style['message']} This estimate suggests the battery may reach end-of-life around cycle <strong>{estimated_failure_cycle}</strong>.
                    </div>
                    <div class="bubble">
                        <div class="bubble-label">Recommended action</div>
                        {style['action']}
                    </div>
                    <div class="bubble">
                        <div class="bubble-label">What the user should focus on</div>
                        Alert level, remaining cycles, and whether maintenance should be planned now.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="chat-panel">
                    <div class="chat-chip">Before Prediction</div>
                    <div class="card-title" style="font-size:1.6rem; margin-bottom:0.55rem;">Run a battery health check</div>
                    <div class="card-copy">
                        Select a battery source and current window, then run the health check to show the alert, remaining cycles, and recommendation here.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if result_payload:
        st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)
        top_result_col, side_result_col = st.columns([1.08, 0.92], gap="large")
        with top_result_col:
            render_result_panel_with_options(
                result_payload=result_payload,
                show_xai=False,
                show_technical=False,
            )
        with side_result_col:
            render_xai_client_summary()
            render_technical_summary_card(result_payload)


def render_training_page() -> None:
    render_section_header(
        "Training Admin",
        "This page is reserved for internal model updates, candidate retraining, and administrative review.",
    )
    st.markdown(
        """
        <div class="chat-panel" style="margin-bottom: 1rem;">
            <div class="chat-chip">Admin Only</div>
            <div class="card-title" style="font-size:1.45rem; margin-bottom:0.45rem;">Retrain the GLU model with a clean battery dataset.</div>
            <div class="card-copy">
                Use this workspace to upload a prepared CSV, configure training behavior, compare candidate runs, and decide whether a new model should replace the approved one.
            </div>
            <div class="bubble" style="margin-top:0.9rem;">
                <div class="bubble-label">Required columns</div>
                <span style="font-family:monospace;">battery_id, cycle, chI, chV, chT, disI, disV, BCt, SOH, RUL</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows: list[dict[str, Any]] = []
    selected_cv: dict[str, Any] = {}
    selected_dir = Path()

    train_col, summary_col = st.columns([0.92, 1.08], gap="large")
    with train_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Dataset And Training Setup</div>
                <div class="section-copy">Upload the new dataset, tune the main training controls, then launch a candidate run.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        training_upload = st.file_uploader("Upload training dataset CSV", type=["csv"], key="candidate_training_upload")
        candidate_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=30, step=1)
        candidate_patience = st.number_input("Early stopping patience", min_value=1, max_value=20, value=10, step=1)
        candidate_cross_validate = st.checkbox("Run leave-one-battery-out cross-validation", value=False)

        st.markdown(
            f"""
            <div class="ui-card" style="margin: 1rem 0 1rem 0;">
                <div class="card-label">Current training plan</div>
                <div class="card-title" style="font-size:1.2rem;">{int(candidate_epochs)} epochs with patience {int(candidate_patience)}</div>
                <div class="card-copy">
                    Cross-validation is <strong>{"enabled" if candidate_cross_validate else "disabled"}</strong>.
                    Use cross-validation when you want a stronger internal comparison before approving a new candidate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Train Candidate GLU Model", key="train_candidate_model"):
            if training_upload is None:
                st.error("Please upload a CSV dataset first.")
            else:
                dataset_path = save_training_upload(training_upload)
                with st.spinner("Training candidate model. This may take a few minutes..."):
                    result = run_candidate_training(
                        dataset_path=dataset_path,
                        epochs=int(candidate_epochs),
                        patience=int(candidate_patience),
                        cross_validate=bool(candidate_cross_validate),
                    )
                if result.returncode == 0:
                    st.success("Candidate training completed.")
                    st.cache_data.clear()
                    with st.expander("Training log"):
                        st.code(result.stdout[-3000:] or "Training completed.")
                else:
                    st.error("Candidate training failed.")
                    with st.expander("Training error log"):
                        st.code(result.stderr[-3000:] or result.stdout[-3000:])

    with summary_col:
        candidate_summaries = load_candidate_summaries()
        if not candidate_summaries:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">Candidate Review</div>
                    <div class="section-copy">No candidate models have been trained yet. Run a training job to populate this review area.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Candidate Review</div>
                <div class="section-copy">Compare recent candidate runs, inspect the current selection, and decide whether to keep the approved model or review the candidate further.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rows = []
        for summary in candidate_summaries[:5]:
            metrics_payload = summary.get("candidate_metrics", {})
            comparison_payload = summary.get("comparison", {})
            rows.append(
                {
                    "candidate": summary.get("candidate_name"),
                    "created": summary.get("modified_time"),
                    "mae": metrics_payload.get("mae"),
                    "rmse": metrics_payload.get("rmse"),
                    "r2": metrics_payload.get("r2"),
                    "recommendation": comparison_payload.get("recommendation"),
                }
            )
        selected_candidate = st.selectbox(
            "Choose candidate run",
            options=candidate_summaries,
            format_func=lambda item: f"{item.get('candidate_name')} ({item.get('modified_time')})",
        )
        selected_metrics = selected_candidate.get("candidate_metrics", {})
        selected_comparison = selected_candidate.get("comparison", {})
        selected_cv = selected_candidate.get("cross_validation", {})
        selected_dir = Path(selected_candidate.get("candidate_dir", ""))
        recommendation = selected_comparison.get("recommendation", "review_manually")
        recommendation_copy = {
            "keep_current_model": "The current approved model still performs better than this candidate.",
            "review_manually": "The metrics are inconclusive, so this run should be reviewed manually before any model swap.",
            "promote_candidate": "This candidate looks stronger than the approved model and may be ready for promotion.",
        }.get(recommendation, "Review this candidate manually before making an approval decision.")

        st.markdown(
            f"""
            <div class="chat-panel" style="margin-top: 1rem; margin-bottom: 1rem;">
                <div class="chat-chip">Selected Candidate</div>
                <div class="card-title" style="font-size:1.4rem; margin-bottom:0.45rem;">{selected_candidate.get('candidate_name')}</div>
                <div class="card-copy" style="margin-bottom:0.9rem;">
                    Created on {selected_candidate.get('modified_time')}. {recommendation_copy}
                </div>
                <div class="bubble">
                    <div class="bubble-label">Recommendation</div>
                    {recommendation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Candidate MAE", f"{selected_metrics.get('mae', 0):.2f}")
        metric_col2.metric("Candidate RMSE", f"{selected_metrics.get('rmse', 0):.2f}")
        metric_col3.metric("Candidate R2", f"{selected_metrics.get('r2', 0):.3f}")

        if selected_comparison.get("rmse_improvement_percent") is not None:
            st.markdown(
                f"""
                <div class="ui-card" style="margin-top: 1rem; margin-bottom: 1rem;">
                    <div class="card-label">Comparison Against Approved Model</div>
                    <div class="card-copy">RMSE improvement vs. reference: <strong>{selected_comparison['rmse_improvement_percent']:.2f}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-card" style="margin-top: 1rem;">
            <div class="section-title">Recent Candidate Runs</div>
            <div class="section-copy">Latest candidate snapshots with the key validation metrics.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    detail_col1, detail_col2 = st.columns([1, 1], gap="large")
    with detail_col1:
        with st.expander("Cross-validation details"):
            if selected_cv.get("available"):
                cv_col1, cv_col2, cv_col3 = st.columns(3)
                cv_col1.metric("CV MAE", f"{selected_cv.get('mae_mean', 0):.2f} +/- {selected_cv.get('mae_std', 0):.2f}")
                cv_col2.metric("CV RMSE", f"{selected_cv.get('rmse_mean', 0):.2f} +/- {selected_cv.get('rmse_std', 0):.2f}")
                cv_col3.metric("CV R2", f"{selected_cv.get('r2_mean', 0):.3f} +/- {selected_cv.get('r2_std', 0):.3f}")
                fold_metrics_path = Path(selected_cv.get("fold_metrics_path", ""))
                if fold_metrics_path.exists():
                    st.dataframe(pd.read_csv(fold_metrics_path), use_container_width=True, hide_index=True)
            else:
                st.info(selected_cv.get("reason", "Cross-validation was not available for this candidate."))

    with detail_col2:
        with st.expander("Candidate output files and plots"):
            plot_col1, plot_col2, plot_col3 = st.columns(3)
            loss_plot = selected_dir / "loss_curve.png"
            true_pred_plot = selected_dir / "true_vs_pred.png"
            error_plot = selected_dir / "prediction_error_over_cycles.png"
            if loss_plot.exists():
                plot_col1.image(str(loss_plot), caption="Training vs Validation Loss")
            if true_pred_plot.exists():
                plot_col2.image(str(true_pred_plot), caption="True RUL vs Predicted RUL")
            if error_plot.exists():
                plot_col3.image(str(error_plot), caption="Prediction Error over Cycles")


def render_insights_page(
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    xai_summary: dict[str, Any],
    xai_permutation: pd.DataFrame,
    xai_ig_feature: pd.DataFrame,
    xai_temporal: pd.DataFrame,
) -> None:
    render_section_header(
        "Insights",
        "Review technical evaluation, explainability outputs, and experiment figures for the approved model.",
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("MAE", f"{metrics['mae']:.2f}")
    metric_col2.metric("RMSE", f"{metrics['rmse']:.2f}")
    metric_col3.metric("R2", f"{metrics['r2']:.3f}")
    metric_col4.metric("Test windows", f"{metrics['test_shape'][0]}")

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        render_section_header("Prediction Error Over Cycles", "Official B6 error trace across the test windows.")
        st.line_chart(predictions.set_index("end_cycle")[["prediction_error"]])
    with plot_col2:
        render_section_header("Top-Level Explainability", "The strongest feature drivers for the approved GLU model.")
        if not xai_ig_feature.empty:
            st.bar_chart(xai_ig_feature.set_index("feature")[["mean_abs_integrated_gradient"]])
        else:
            st.info("Integrated gradients feature table was not found.")

    temporal_col, permutation_col = st.columns(2)
    with temporal_col:
        render_section_header("Temporal Importance", "Which step in the 10-cycle window matters most.")
        if not xai_temporal.empty:
            st.line_chart(xai_temporal.set_index("window_step")[["mean_abs_integrated_gradient"]])
        else:
            st.info("Temporal XAI table was not found.")

    with permutation_col:
        render_section_header("Permutation Importance", "Global importance based on the degradation in performance when a feature is shuffled.")
        if not xai_permutation.empty:
            st.dataframe(
                xai_permutation[["feature", "rmse_increase", "mae_increase", "r2_drop"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Permutation importance table was not found.")

    if xai_summary:
        st.markdown(
            """
            <div class="footer-note">
                XAI confirms the model leans heavily on capacity and health signals, especially near the most recent cycles in the window.
            </div>
            """,
            unsafe_allow_html=True,
        )

    figure_col1, figure_col2, figure_col3, figure_col4 = st.columns(4)
    if XAI_PERMUTATION_FIGURE.exists():
        figure_col1.image(str(XAI_PERMUTATION_FIGURE), caption="Permutation Importance")
    if XAI_IG_FEATURE_FIGURE.exists():
        figure_col2.image(str(XAI_IG_FEATURE_FIGURE), caption="Integrated Gradients Features")
    if XAI_TEMPORAL_FIGURE.exists():
        figure_col3.image(str(XAI_TEMPORAL_FIGURE), caption="Temporal Importance")
    if XAI_HEATMAP_FIGURE.exists():
        figure_col4.image(str(XAI_HEATMAP_FIGURE), caption="Feature-Time Heatmap")

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    if LOSS_CURVE_PATH.exists():
        exp_col1.image(str(LOSS_CURVE_PATH), caption="Training vs Validation Loss")
    if TRUE_VS_PRED_PATH.exists():
        exp_col2.image(str(TRUE_VS_PRED_PATH), caption="True RUL vs Predicted RUL")
    if ERROR_OVER_CYCLES_PATH.exists():
        exp_col3.image(str(ERROR_OVER_CYCLES_PATH), caption="Prediction Error Over Cycles")


metrics = load_metrics()
predictions = load_predictions()
processed_cycle_data = load_processed_cycle_data()
xai_summary = load_xai_summary()
xai_permutation = load_xai_table(XAI_PERMUTATION_PATH)
xai_ig_feature = load_xai_table(XAI_IG_FEATURE_PATH)
xai_temporal = load_xai_table(XAI_TEMPORAL_PATH)

page = get_current_page()

if page == "Overview":
    st.markdown(
        """
        <style>
        html, body, .stApp {
            height: 100vh;
            overflow: hidden;
        }

        .main .block-container {
            height: calc(100vh - 1rem);
            overflow: hidden;
            padding-top: 0.9rem;
            padding-bottom: 0.1rem;
        }

        .app-layout,
        .content-shell {
            height: 100%;
            overflow: hidden;
        }

        .dashboard-shell {
            height: calc(100vh - 2rem);
            padding: 0.85rem;
            overflow: hidden;
        }

        .hero-panel,
        .chat-panel {
            padding: 1rem;
        }

        .hero-title {
            font-size: clamp(1.65rem, 2.45vw, 2.55rem);
            margin-bottom: 0.45rem;
        }

        .hero-copy {
            font-size: 0.88rem;
            line-height: 1.46;
        }

        .progress-banner {
            margin-top: 0.75rem;
            padding: 0.62rem 0.82rem;
        }

        .bubble {
            padding: 0.72rem 0.82rem;
            margin-bottom: 0.5rem;
        }

        .footer-note {
            margin-top: 0.55rem;
            padding: 0.72rem 0.9rem;
        }

        .chat-chip,
        .hero-kicker {
            margin-bottom: 0.7rem;
        }

        .progress-meta {
            margin-bottom: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="app-layout">', unsafe_allow_html=True)
rail_col, content_col = st.columns([0.12, 0.88], gap="large")

with rail_col:
    render_visual_rail(page)

with content_col:
    st.markdown('<div class="content-shell">', unsafe_allow_html=True)
    api_url = st.session_state.get("global_api_url", DEFAULT_API_URL)
    if page == "Overview":
        render_overview_page(metrics, predictions, processed_cycle_data)
    elif page == "Prediction":
        render_prediction_page(api_url, metrics, predictions, processed_cycle_data)
    elif page == "Training":
        render_training_page()
    else:
        render_insights_page(
            metrics,
            predictions,
            xai_summary,
            xai_permutation,
            xai_ig_feature,
            xai_temporal,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
