import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
RESULTS_DIR = WORKSPACE / "results"
OUTPUT_DIR = RESULTS_DIR / "glu_summary_plots"

MODEL_RESULTS = [
    ("Small GLU", "glu_small", 401),
    ("Medium GLU", "glu_medium", 1185),
    ("Large GLU", "glu_large", 7809),
    ("Medium GLU + Pooling", "glu_medium_pooling", 1697),
    ("Two GLU Blocks + Pooling", "glu_two_blocks_pooling", 3809),
    ("Two GLU Blocks + Residual + Pooling", "glu_two_blocks_residual_pooling", 3809),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_metrics() -> pd.DataFrame:
    rows = []
    for model_name, folder_name, trainable_params in MODEL_RESULTS:
        metrics_path = RESULTS_DIR / folder_name / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        rows.append(
            {
                "model": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "trainable_params": trainable_params,
            }
        )
    return pd.DataFrame(rows)


def plot_metric_bars(df: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: Path, lower_is_better: bool) -> None:
    plot_df = df.sort_values(metric, ascending=lower_is_better)
    colors = ["#2b6cb0" if i == 0 else "#8aa8c8" for i in range(len(plot_df))]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(plot_df["model"], plot_df[metric], color=colors)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)

    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}" if metric == "r2" else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_model_metric_grouped(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.copy()
    plot_df["r2_scaled"] = plot_df["r2"] * 100

    x = range(len(plot_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar([i - width for i in x], plot_df["mae"], width=width, label="MAE")
    plt.bar(x, plot_df["rmse"], width=width, label="RMSE")
    plt.bar([i + width for i in x], plot_df["r2_scaled"], width=width, label="R² x 100")
    plt.xticks(list(x), plot_df["model"], rotation=25, ha="right")
    plt.ylabel("Value")
    plt.title("GLU Model Comparison: MAE, RMSE, and R²")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_params_vs_rmse(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["trainable_params"], df["rmse"], s=90)
    for _, row in df.iterrows():
        plt.annotate(row["model"], (row["trainable_params"], row["rmse"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    plt.xlabel("Trainable Parameters")
    plt.ylabel("RMSE")
    plt.title("Model Size vs RMSE")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cv_folds(cv_df: pd.DataFrame, output_path: Path) -> None:
    labels = [f"Test {battery}" for battery in cv_df["test_batteries"]]
    x = range(len(cv_df))
    width = 0.28

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], cv_df["mae"], width=width, label="MAE")
    plt.bar([i + width / 2 for i in x], cv_df["rmse"], width=width, label="RMSE")
    plt.xticks(list(x), labels)
    plt.ylabel("Cycles")
    plt.title("Leave-One-Battery-Out CV: MAE and RMSE by Test Battery")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cv_r2(cv_df: pd.DataFrame, output_path: Path) -> None:
    labels = [f"Test {battery}" for battery in cv_df["test_batteries"]]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, cv_df["r2"], color="#2b6cb0")
    plt.ylabel("R²")
    plt.title("Leave-One-Battery-Out CV: R² by Test Battery")
    plt.ylim(0, max(1.0, float(cv_df["r2"].max()) + 0.1))
    plt.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cv_aggregate(aggregate: dict, output_path: Path) -> None:
    metrics = ["MAE", "RMSE", "R²"]
    means = [aggregate["mae_mean"], aggregate["rmse_mean"], aggregate["r2_mean"]]
    stds = [aggregate["mae_std"], aggregate["rmse_std"], aggregate["r2_std"]]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(metrics, means, yerr=stds, capsize=8, color=["#8aa8c8", "#5f8db8", "#2b6cb0"])
    plt.ylabel("Mean ± Std")
    plt.title("Cross-Validation Aggregate Metrics")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}" if value < 1 else f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_df = load_model_metrics()
    model_df.to_csv(OUTPUT_DIR / "glu_model_comparison_metrics.csv", index=False)

    plot_metric_bars(model_df, "mae", "MAE", "GLU Model Comparison: MAE Lower Is Better", OUTPUT_DIR / "model_mae_comparison.png", True)
    plot_metric_bars(model_df, "rmse", "RMSE", "GLU Model Comparison: RMSE Lower Is Better", OUTPUT_DIR / "model_rmse_comparison.png", True)
    plot_metric_bars(model_df, "r2", "R²", "GLU Model Comparison: R² Higher Is Better", OUTPUT_DIR / "model_r2_comparison.png", False)
    plot_model_metric_grouped(model_df, OUTPUT_DIR / "model_metrics_grouped.png")
    plot_params_vs_rmse(model_df, OUTPUT_DIR / "model_size_vs_rmse.png")

    cv_dir = RESULTS_DIR / "glu_cv_leave_one_battery_out"
    cv_metrics_path = cv_dir / "fold_metrics.csv"
    cv_aggregate_path = cv_dir / "aggregate_metrics.json"
    if cv_metrics_path.exists() and cv_aggregate_path.exists():
        cv_df = pd.read_csv(cv_metrics_path)
        aggregate = load_json(cv_aggregate_path)
        cv_df.to_csv(OUTPUT_DIR / "cv_fold_metrics.csv", index=False)
        plot_cv_folds(cv_df, OUTPUT_DIR / "cv_fold_mae_rmse.png")
        plot_cv_r2(cv_df, OUTPUT_DIR / "cv_fold_r2.png")
        plot_cv_aggregate(aggregate, OUTPUT_DIR / "cv_aggregate_metrics.png")

    print(f"Saved summary plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
