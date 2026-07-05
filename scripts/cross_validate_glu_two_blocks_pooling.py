from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


WORKSPACE = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = WORKSPACE / "data" / "raw" / "Battery_dataset.csv"
RESULTS_DIR = WORKSPACE / "results" / "glu_cv_leave_one_battery_out"

FEATURE_COLUMNS = ["chI", "chV", "chT", "disI", "disV", "BCt", "SOH"]
WINDOW_SIZE = 10
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001
HIDDEN_DIM = 32
DROPOUT = 0.15
DENSE_UNITS = 16


@dataclass
class FoldConfig:
    fold: int
    train_batteries: list[str]
    validation_batteries: list[str]
    test_batteries: list[str]
    window_size: int
    feature_columns: list[str]
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    hidden_dim: int
    dropout: float
    dense_units: int
    seed: int


class GLUBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.projection = tf.keras.layers.Dense(hidden_dim * 2, name="glu_projection")

    def call(self, inputs):
        projected = self.projection(inputs)
        a, b = tf.split(projected, num_or_size_splits=2, axis=-1)
        return a * tf.sigmoid(b)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def fit_scaler(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    params: dict[str, tuple[float, float]] = {}
    for column in FEATURE_COLUMNS:
        numeric = pd.to_numeric(train_df[column], errors="raise")
        params[column] = (float(numeric.min()), float(numeric.max()))
    return params


def apply_scaler(df: pd.DataFrame, params: dict[str, tuple[float, float]]) -> pd.DataFrame:
    scaled = df.copy()
    for column, (column_min, column_max) in params.items():
        numeric = pd.to_numeric(scaled[column], errors="raise")
        if column_max == column_min:
            scaled[column] = 0.0
        else:
            scaled[column] = (numeric - column_min) / (column_max - column_min)
    scaled["RUL"] = pd.to_numeric(scaled["RUL"], errors="raise")
    return scaled


def aggregate_to_cycles(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["battery_id", "cycle", *FEATURE_COLUMNS, "RUL"]
    aggregated = (
        df[columns]
        .groupby(["battery_id", "cycle"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["battery_id", "cycle"])
        .reset_index(drop=True)
    )
    return aggregated


def make_windows(cycle_df: pd.DataFrame, battery_ids: list[str]):
    windows: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, float | str]] = []

    for battery_id in battery_ids:
        battery_df = (
            cycle_df[cycle_df["battery_id"] == battery_id]
            .sort_values("cycle")
            .reset_index(drop=True)
        )
        if len(battery_df) < WINDOW_SIZE:
            continue

        features = battery_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        rul = battery_df["RUL"].to_numpy(dtype=np.float32)
        cycles = battery_df["cycle"].to_numpy()

        for start in range(0, len(battery_df) - WINDOW_SIZE + 1):
            end = start + WINDOW_SIZE
            windows.append(features[start:end])
            targets.append(float(rul[end - 1]))
            metadata.append(
                {
                    "battery_id": battery_id,
                    "start_cycle": float(cycles[start]),
                    "end_cycle": float(cycles[end - 1]),
                    "target_RUL": float(rul[end - 1]),
                }
            )

    if not windows:
        raise ValueError(f"No windows were created for batteries: {battery_ids}")

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.DataFrame(metadata),
    )


def build_fold_data(raw_df: pd.DataFrame, train_ids: list[str], val_ids: list[str], test_ids: list[str]):
    train_raw = raw_df[raw_df["battery_id"].isin(train_ids)]
    scaler = fit_scaler(train_raw)
    scaled = apply_scaler(raw_df, scaler)
    cycle_df = aggregate_to_cycles(scaled)

    X_train, y_train, train_metadata = make_windows(cycle_df, train_ids)
    X_val, y_val, val_metadata = make_windows(cycle_df, val_ids)
    X_test, y_test, test_metadata = make_windows(cycle_df, test_ids)
    return X_train, y_train, X_val, y_val, X_test, y_test, train_metadata, val_metadata, test_metadata


def build_model(sequence_length: int, input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(sequence_length, input_dim), name="cycle_window")
    x = GLUBlock(HIDDEN_DIM, name="glu_block_1")(inputs)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout_1")(x)
    x = GLUBlock(HIDDEN_DIM, name="glu_block_2")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout_2")(x)

    last_step = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="average_pooling")(x)
    x = tf.keras.layers.Concatenate(name="last_plus_average")([last_step, avg_pool])

    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="regression_hidden_1")(x)
    x = tf.keras.layers.Dense(DENSE_UNITS // 2, activation="relu", name="regression_hidden_2")(x)
    outputs = tf.keras.layers.Dense(1, name="rul_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="glu_two_blocks_pooling_regressor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def plot_losses(history, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.75)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_prediction_error(metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    errors = y_pred - y_true
    plt.figure(figsize=(9, 5))
    for battery_id, group in metadata.assign(error=errors).groupby("battery_id"):
        plt.plot(group["end_cycle"], group["error"], marker="o", linewidth=1.5, markersize=3, label=battery_id)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("End Cycle of Window")
    plt.ylabel("Prediction Error (Predicted - True)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_json(data: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def make_folds(battery_ids: list[str]) -> list[tuple[list[str], list[str], list[str]]]:
    validation_for_test = {
        "B5": "B6",
        "B6": "B18",
        "B7": "B5",
        "B18": "B7",
    }
    folds = []
    for test_id in battery_ids:
        val_id = validation_for_test.get(test_id)
        if val_id is None or val_id == test_id:
            remaining = [battery_id for battery_id in battery_ids if battery_id != test_id]
            val_id = remaining[-1]
        train_ids = [battery_id for battery_id in battery_ids if battery_id not in {test_id, val_id}]
        folds.append((train_ids, [val_id], [test_id]))
    return folds


def train_fold(
    fold_index: int,
    raw_df: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> dict[str, float | int | str]:
    fold_dir = RESULTS_DIR / f"fold_{fold_index}_{test_ids[0]}_test"
    fold_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED + fold_index)
    tf.keras.backend.clear_session()

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_metadata,
        val_metadata,
        test_metadata,
    ) = build_fold_data(raw_df, train_ids, val_ids, test_ids)

    model = build_model(sequence_length=X_train.shape[1], input_dim=X_train.shape[2])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
    )

    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    test_eval = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)

    metrics = {
        "fold": fold_index,
        "train_batteries": ",".join(train_ids),
        "validation_batteries": ",".join(val_ids),
        "test_batteries": ",".join(test_ids),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "test_loss_mse": float(test_eval[0]),
        "test_metric_mae": float(test_eval[1]),
        "train_windows": int(len(y_train)),
        "validation_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
    }

    config = FoldConfig(
        fold=fold_index,
        train_batteries=train_ids,
        validation_batteries=val_ids,
        test_batteries=test_ids,
        window_size=WINDOW_SIZE,
        feature_columns=FEATURE_COLUMNS,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        dense_units=DENSE_UNITS,
        seed=SEED + fold_index,
    )

    predictions_df = test_metadata.copy()
    predictions_df["true_RUL"] = y_test
    predictions_df["predicted_RUL"] = y_pred
    predictions_df["prediction_error"] = predictions_df["predicted_RUL"] - predictions_df["true_RUL"]

    save_json(metrics, fold_dir / "metrics.json")
    save_json(asdict(config), fold_dir / "config.json")
    pd.DataFrame(history.history).to_csv(fold_dir / "history.csv", index=False)
    train_metadata.to_csv(fold_dir / "train_window_metadata.csv", index=False)
    val_metadata.to_csv(fold_dir / "validation_window_metadata.csv", index=False)
    predictions_df.to_csv(fold_dir / "test_predictions.csv", index=False)
    with (fold_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))
    model.save(fold_dir / "model.keras")

    plot_losses(history, fold_dir / "loss_curve.png", f"Fold {fold_index}: Training vs Validation Loss")
    plot_true_vs_pred(y_test, y_pred, fold_dir / "true_vs_pred.png", f"Fold {fold_index}: True RUL vs Predicted RUL")
    plot_prediction_error(test_metadata, y_test, y_pred, fold_dir / "prediction_error_over_cycles.png", f"Fold {fold_index}: Prediction Error over Cycles")

    return metrics


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    battery_ids = sorted(raw_df["battery_id"].unique(), key=lambda value: int(str(value).replace("B", "")))
    folds = make_folds(battery_ids)

    all_metrics = []
    for fold_index, (train_ids, val_ids, test_ids) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_index}: train={train_ids}, val={val_ids}, test={test_ids} ===")
        all_metrics.append(train_fold(fold_index, raw_df, train_ids, val_ids, test_ids))

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)

    aggregate = {
        "folds": len(summary_df),
        "mae_mean": float(summary_df["mae"].mean()),
        "mae_std": float(summary_df["mae"].std(ddof=1)),
        "rmse_mean": float(summary_df["rmse"].mean()),
        "rmse_std": float(summary_df["rmse"].std(ddof=1)),
        "r2_mean": float(summary_df["r2"].mean()),
        "r2_std": float(summary_df["r2"].std(ddof=1)),
        "feature_columns": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "model": "two_glu_blocks_plus_last_timestep_and_average_pooling",
    }
    save_json(aggregate, RESULTS_DIR / "aggregate_metrics.json")
    print("\nCross-validation fold metrics:")
    print(summary_df[["fold", "train_batteries", "validation_batteries", "test_batteries", "mae", "rmse", "r2"]].to_string(index=False))
    print("\nAggregate metrics:")
    print(json.dumps(aggregate, indent=2))
    print(f"\nSaved cross-validation results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
