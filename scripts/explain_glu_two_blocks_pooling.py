import argparse
import json
import math
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_glu_two_blocks_pooling import build_model


WINDOWS_PATH = WORKSPACE / "data" / "processed" / "windows_w10.npz"
MODEL_PATH = WORKSPACE / "results" / "glu_two_blocks_pooling" / "glu_two_blocks_pooling_model.keras"
TEST_METADATA_PATH = WORKSPACE / "data" / "processed" / "test_window_metadata_w10.csv"
OUTPUT_DIR = WORKSPACE / "results" / "xai_glu_two_blocks_pooling"

SEED = 42
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain the approved GLU two-block pooling model for Battery RUL prediction."
    )
    parser.add_argument("--windows-path", type=Path, default=WINDOWS_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=TEST_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--integrated-gradient-steps", type=int, default=50)
    return parser.parse_args()


def load_windows(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "X_train": data["X_train"].astype(np.float32),
        "X_test": data["X_test"].astype(np.float32),
        "y_test": data["y_test"].astype(np.float32),
        "feature_columns": data["feature_columns"].tolist(),
        "window_size": int(data["window_size"]),
    }


def load_glu_model(path: Path, sequence_length: int, input_dim: int) -> tf.keras.Model:
    model = build_model(sequence_length=sequence_length, input_dim=input_dim)
    model.load_weights(path)
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def predict(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, batch_size=BATCH_SIZE, verbose=0).reshape(-1)


def sample_explanation_set(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(X) <= max_samples:
        indexes = np.arange(len(X))
    else:
        indexes = np.linspace(0, len(X) - 1, max_samples, dtype=int)
    return X[indexes], y[indexes], indexes


def permutation_importance(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_pred = predict(model, X)
    baseline_metrics = regression_metrics(y, baseline_pred)
    rows = []

    for feature_idx, feature_name in enumerate(feature_columns):
        repeated_metrics = []
        for _ in range(repeats):
            X_permuted = X.copy()
            shuffled_values = X_permuted[:, :, feature_idx].reshape(-1).copy()
            rng.shuffle(shuffled_values)
            X_permuted[:, :, feature_idx] = shuffled_values.reshape(
                X_permuted.shape[0],
                X_permuted.shape[1],
            )
            permuted_pred = predict(model, X_permuted)
            repeated_metrics.append(regression_metrics(y, permuted_pred))

        rmse_values = np.array([item["rmse"] for item in repeated_metrics])
        mae_values = np.array([item["mae"] for item in repeated_metrics])
        r2_values = np.array([item["r2"] for item in repeated_metrics])
        rows.append(
            {
                "feature": feature_name,
                "baseline_rmse": baseline_metrics["rmse"],
                "permuted_rmse_mean": float(rmse_values.mean()),
                "permuted_rmse_std": float(rmse_values.std(ddof=0)),
                "rmse_increase": float(rmse_values.mean() - baseline_metrics["rmse"]),
                "baseline_mae": baseline_metrics["mae"],
                "permuted_mae_mean": float(mae_values.mean()),
                "mae_increase": float(mae_values.mean() - baseline_metrics["mae"]),
                "baseline_r2": baseline_metrics["r2"],
                "permuted_r2_mean": float(r2_values.mean()),
                "r2_drop": float(baseline_metrics["r2"] - r2_values.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False)


def integrated_gradients(
    model: tf.keras.Model,
    X: np.ndarray,
    baseline: np.ndarray,
    steps: int,
) -> np.ndarray:
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    baseline_tensor = tf.convert_to_tensor(baseline, dtype=tf.float32)
    delta = X_tensor - baseline_tensor
    gradient_sum = tf.zeros_like(X_tensor)

    for alpha in alphas:
        interpolated = baseline_tensor + alpha * delta
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = model(interpolated, training=False)
        gradient_sum += tape.gradient(predictions, interpolated)

    average_gradients = gradient_sum / tf.cast(len(alphas), tf.float32)
    return (delta * average_gradients).numpy()


def attribution_tables(
    attributions: np.ndarray,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    absolute_attr = np.abs(attributions)
    feature_scores = absolute_attr.mean(axis=(0, 1))
    temporal_scores = absolute_attr.mean(axis=(0, 2))
    heatmap_scores = absolute_attr.mean(axis=0)

    feature_df = pd.DataFrame(
        {"feature": feature_columns, "mean_abs_integrated_gradient": feature_scores}
    ).sort_values("mean_abs_integrated_gradient", ascending=False)
    temporal_df = pd.DataFrame(
        {
            "window_step": np.arange(1, len(temporal_scores) + 1),
            "mean_abs_integrated_gradient": temporal_scores,
        }
    )
    heatmap_df = pd.DataFrame(heatmap_scores, columns=feature_columns)
    heatmap_df.insert(0, "window_step", np.arange(1, len(heatmap_df) + 1))
    return feature_df, temporal_df, heatmap_df


def plot_bar(df: pd.DataFrame, x_column: str, y_column: str, title: str, output_path: Path) -> None:
    ordered = df.sort_values(y_column, ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(ordered[x_column], ordered[y_column])
    plt.xlabel(y_column.replace("_", " ").title())
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_temporal_importance(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["window_step"], df["mean_abs_integrated_gradient"], marker="o")
    plt.xlabel("Window Step")
    plt.ylabel("Mean Absolute Integrated Gradient")
    plt.title("GLU XAI: Temporal Importance Across 10-Cycle Window")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_heatmap(heatmap_df: pd.DataFrame, feature_columns: list[str], output_path: Path) -> None:
    values = heatmap_df[feature_columns].to_numpy()
    plt.figure(figsize=(9, 5))
    image = plt.imshow(values, aspect="auto", cmap="viridis")
    plt.colorbar(image, label="Mean Absolute Integrated Gradient")
    plt.xticks(np.arange(len(feature_columns)), feature_columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(heatmap_df)), heatmap_df["window_step"])
    plt.xlabel("Feature")
    plt.ylabel("Window Step")
    plt.title("GLU XAI: Feature Importance by Window Step")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_json(data: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    windows = load_windows(args.windows_path)
    metadata = pd.read_csv(args.metadata_path)

    X_test, y_test, selected_indexes = sample_explanation_set(
        windows["X_test"],
        windows["y_test"],
        args.max_samples,
    )
    model = load_glu_model(
        args.model_path,
        sequence_length=X_test.shape[1],
        input_dim=X_test.shape[2],
    )
    selected_metadata = metadata.iloc[selected_indexes].reset_index(drop=True)

    y_pred = predict(model, X_test)
    baseline_metrics = regression_metrics(y_test, y_pred)

    permutation_df = permutation_importance(
        model,
        X_test,
        y_test,
        windows["feature_columns"],
        args.permutation_repeats,
        SEED,
    )

    baseline_window = windows["X_train"].mean(axis=0, keepdims=True)
    baseline_window = np.repeat(baseline_window, repeats=len(X_test), axis=0)
    attributions = integrated_gradients(
        model,
        X_test,
        baseline_window,
        args.integrated_gradient_steps,
    )
    feature_df, temporal_df, heatmap_df = attribution_tables(attributions, windows["feature_columns"])

    permutation_df.to_csv(args.output_dir / "permutation_importance.csv", index=False)
    feature_df.to_csv(args.output_dir / "integrated_gradients_feature_importance.csv", index=False)
    temporal_df.to_csv(args.output_dir / "integrated_gradients_temporal_importance.csv", index=False)
    heatmap_df.to_csv(args.output_dir / "integrated_gradients_heatmap.csv", index=False)
    selected_metadata.assign(true_RUL=y_test, predicted_RUL=y_pred).to_csv(
        args.output_dir / "explained_samples.csv",
        index=False,
    )

    plot_bar(
        permutation_df,
        "feature",
        "rmse_increase",
        "GLU XAI: Permutation Feature Importance",
        args.output_dir / "permutation_importance.png",
    )
    plot_bar(
        feature_df,
        "feature",
        "mean_abs_integrated_gradient",
        "GLU XAI: Integrated Gradients Feature Importance",
        args.output_dir / "integrated_gradients_feature_importance.png",
    )
    plot_temporal_importance(temporal_df, args.output_dir / "integrated_gradients_temporal_importance.png")
    plot_heatmap(heatmap_df, windows["feature_columns"], args.output_dir / "integrated_gradients_heatmap.png")

    summary = {
        "model_path": str(args.model_path),
        "windows_path": str(args.windows_path),
        "metadata_path": str(args.metadata_path),
        "output_dir": str(args.output_dir),
        "samples_explained": int(len(X_test)),
        "window_size": windows["window_size"],
        "feature_columns": windows["feature_columns"],
        "baseline_metrics": baseline_metrics,
        "top_permutation_features": permutation_df.head(3).to_dict(orient="records"),
        "top_integrated_gradient_features": feature_df.head(3).to_dict(orient="records"),
    }
    save_json(summary, args.output_dir / "xai_summary.json")

    print(json.dumps(summary, indent=2))
    print(f"Saved XAI outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
