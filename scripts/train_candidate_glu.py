from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


WORKSPACE = Path(__file__).resolve().parents[1]
PRODUCTION_RESULTS_DIR = WORKSPACE / "results" / "glu_two_blocks_pooling"
PRODUCTION_MODEL_PATH = PRODUCTION_RESULTS_DIR / "glu_two_blocks_pooling_model.keras"
PRODUCTION_METRICS_PATH = PRODUCTION_RESULTS_DIR / "metrics.json"
MODEL_REGISTRY_DIR = WORKSPACE / "model_registry"

FEATURE_COLUMNS = ["chI", "chV", "chT", "disI", "disV", "BCt", "SOH"]
REQUIRED_COLUMNS = ["battery_id", "cycle", *FEATURE_COLUMNS, "RUL"]
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
class CandidateConfig:
    uploaded_dataset_path: str
    output_dir: str
    split_strategy: str
    feature_columns: list[str]
    window_size: int
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    hidden_dim: int
    dropout: float
    dense_units: int
    seed: int


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a candidate Two GLU Blocks + Pooling model from an uploaded battery CSV."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="CSV dataset containing battery_id, cycle, seven feature columns, and RUL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output folder. Defaults to model_registry/candidates/candidate_<timestamp>.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run leave-one-battery-out cross-validation after candidate training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


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
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="candidate_glu_two_blocks_pooling")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def validate_dataset(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    for column in ["cycle", *FEATURE_COLUMNS, "RUL"]:
        pd.to_numeric(df[column], errors="raise")

    if df[REQUIRED_COLUMNS].isna().any().any():
        missing_counts = df[REQUIRED_COLUMNS].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Dataset contains missing values: {missing_counts}")

    cycle_counts = df.groupby("battery_id")["cycle"].nunique()
    short_batteries = cycle_counts[cycle_counts < WINDOW_SIZE]
    if not short_batteries.empty:
        raise ValueError(
            "Every battery must have at least "
            f"{WINDOW_SIZE} cycles. Short batteries: {short_batteries.to_dict()}"
        )


def aggregate_to_cycles(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.loc[:, REQUIRED_COLUMNS].copy()
    for column in ["cycle", *FEATURE_COLUMNS, "RUL"]:
        clean[column] = pd.to_numeric(clean[column], errors="raise")

    return (
        clean.groupby(["battery_id", "cycle"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["battery_id", "cycle"])
        .reset_index(drop=True)
    )


def split_batteries(cycle_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    battery_ids = np.array(sorted(cycle_df["battery_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(battery_ids)

    if len(battery_ids) >= 3:
        test_count = max(1, int(round(len(battery_ids) * 0.15)))
        val_count = max(1, int(round(len(battery_ids) * 0.15)))
        train_count = len(battery_ids) - val_count - test_count
        if train_count < 1:
            train_count = 1
            val_count = max(1, len(battery_ids) - train_count - test_count)

        train_ids = set(battery_ids[:train_count])
        val_ids = set(battery_ids[train_count : train_count + val_count])
        test_ids = set(battery_ids[train_count + val_count :])

        train_df = cycle_df[cycle_df["battery_id"].isin(train_ids)].copy()
        val_df = cycle_df[cycle_df["battery_id"].isin(val_ids)].copy()
        test_df = cycle_df[cycle_df["battery_id"].isin(test_ids)].copy()
        return train_df, val_df, test_df, "battery_wise_70_15_15"

    windows_df = cycle_df.copy().reset_index(drop=True)
    first_split = int(len(windows_df) * 0.70)
    second_split = int(len(windows_df) * 0.85)
    train_df = windows_df.iloc[:first_split].copy()
    val_df = windows_df.iloc[first_split:second_split].copy()
    test_df = windows_df.iloc[second_split:].copy()
    return train_df, val_df, test_df, "chronological_single_or_two_battery"


def validate_split_has_windows(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    split_frames = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }
    invalid: dict[str, dict[str, int]] = {}
    for split_name, split_df in split_frames.items():
        cycle_counts = split_df.groupby("battery_id")["cycle"].nunique()
        usable = cycle_counts[cycle_counts >= WINDOW_SIZE]
        if usable.empty:
            invalid[split_name] = cycle_counts.to_dict()

    if invalid:
        raise ValueError(
            "The uploaded dataset is too small after splitting to create 10-cycle "
            "windows for every required split. Use a larger training dataset with "
            f"at least {WINDOW_SIZE} cycles in train, validation, and test. "
            f"Problematic split cycle counts: {invalid}"
        )


def fit_scaler(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    params: dict[str, tuple[float, float]] = {}
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(train_df[column], errors="raise")
        params[column] = (float(values.min()), float(values.max()))
    return params


def apply_scaler(df: pd.DataFrame, params: dict[str, tuple[float, float]]) -> pd.DataFrame:
    scaled = df.copy()
    for column, (column_min, column_max) in params.items():
        values = pd.to_numeric(scaled[column], errors="raise")
        if column_max == column_min:
            scaled[column] = 0.0
        else:
            scaled[column] = (values - column_min) / (column_max - column_min)
    scaled["RUL"] = pd.to_numeric(scaled["RUL"], errors="raise")
    return scaled


def create_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X: list[np.ndarray] = []
    y: list[float] = []
    metadata_rows: list[dict[str, Any]] = []

    for battery_id, group in df.sort_values(["battery_id", "cycle"]).groupby("battery_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        if len(group) < WINDOW_SIZE:
            continue

        for start in range(0, len(group) - WINDOW_SIZE + 1):
            window = group.iloc[start : start + WINDOW_SIZE]
            X.append(window[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            target = float(window["RUL"].iloc[-1])
            y.append(target)
            metadata_rows.append(
                {
                    "battery_id": battery_id,
                    "start_cycle": int(window["cycle"].iloc[0]),
                    "end_cycle": int(window["cycle"].iloc[-1]),
                    "target_RUL": target,
                }
            )

    if not X:
        raise ValueError("No valid 10-cycle windows could be created from the dataset.")

    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.float32), pd.DataFrame(metadata_rows)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_losses(history, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Candidate GLU: Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.75)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Candidate GLU: True RUL vs Predicted RUL")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_prediction_error(metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    plot_df = metadata.copy()
    plot_df["error"] = y_pred - y_true
    plt.figure(figsize=(9, 5))
    for battery_id, group in plot_df.groupby("battery_id"):
        plt.plot(group["end_cycle"], group["error"], marker="o", linewidth=1.5, markersize=3, label=battery_id)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("End Cycle of Window")
    plt.ylabel("Prediction Error (Predicted - True)")
    plt.title("Candidate GLU: Prediction Error over Cycles")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History, np.ndarray]:
    model = build_model(sequence_length=X_train.shape[1], input_dim=X_train.shape[2])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
    )
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
    return model, history, y_pred


def evaluate_production_model(X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    if not PRODUCTION_MODEL_PATH.exists():
        return {
            "available": False,
            "reason": f"Production model not found: {PRODUCTION_MODEL_PATH}",
        }

    model = tf.keras.models.load_model(
        PRODUCTION_MODEL_PATH,
        custom_objects={"GLUBlock": GLUBlock},
        compile=False,
        safe_mode=False,
    )
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    return {
        "available": True,
        "metrics": metrics_dict(y_test, y_pred),
    }


def choose_validation_batteries(train_battery_ids: list[str], seed: int) -> tuple[set[str], set[str]]:
    if len(train_battery_ids) < 2:
        raise ValueError("Cross-validation requires at least three batteries.")

    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(train_battery_ids))
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * 0.20)))
    validation_ids = set(shuffled[:validation_count])
    final_train_ids = set(shuffled[validation_count:])

    if not final_train_ids:
        final_train_ids = {shuffled[-1]}
        validation_ids = set(shuffled[:-1])

    return final_train_ids, validation_ids


def run_leave_one_battery_out_cv(
    cycle_df: pd.DataFrame,
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    battery_ids = sorted(cycle_df["battery_id"].unique().tolist())
    if len(battery_ids) < 3:
        return {
            "available": False,
            "reason": "Cross-validation requires at least three batteries.",
        }

    cv_dir = output_dir / "cross_validation"
    cv_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []

    for fold_number, test_battery in enumerate(battery_ids, start=1):
        train_candidates = [battery for battery in battery_ids if battery != test_battery]
        train_ids, val_ids = choose_validation_batteries(train_candidates, seed + fold_number)

        train_raw = cycle_df[cycle_df["battery_id"].isin(train_ids)].copy()
        val_raw = cycle_df[cycle_df["battery_id"].isin(val_ids)].copy()
        test_raw = cycle_df[cycle_df["battery_id"] == test_battery].copy()
        validate_split_has_windows(train_raw, val_raw, test_raw)

        scaler = fit_scaler(train_raw)
        train_df = apply_scaler(train_raw, scaler)
        val_df = apply_scaler(val_raw, scaler)
        test_df = apply_scaler(test_raw, scaler)

        X_train, y_train, _ = create_windows(train_df)
        X_val, y_val, _ = create_windows(val_df)
        X_test, y_test, test_metadata = create_windows(test_df)

        _, _, y_pred = train_and_predict(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
        )
        fold_metrics = metrics_dict(y_test, y_pred)
        fold_row = {
            "fold": fold_number,
            "test_battery": test_battery,
            "train_batteries": ",".join(sorted(train_ids)),
            "validation_batteries": ",".join(sorted(val_ids)),
            **fold_metrics,
        }
        fold_rows.append(fold_row)

        fold_predictions = test_metadata.copy()
        fold_predictions["true_RUL"] = y_test
        fold_predictions["predicted_RUL"] = y_pred
        fold_predictions["prediction_error"] = fold_predictions["predicted_RUL"] - fold_predictions["true_RUL"]
        fold_predictions.to_csv(cv_dir / f"fold_{fold_number}_{test_battery}_predictions.csv", index=False)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(cv_dir / "fold_metrics.csv", index=False)
    aggregate = {
        "available": True,
        "fold_count": int(len(fold_df)),
        "mae_mean": float(fold_df["mae"].mean()),
        "mae_std": float(fold_df["mae"].std(ddof=0)),
        "rmse_mean": float(fold_df["rmse"].mean()),
        "rmse_std": float(fold_df["rmse"].std(ddof=0)),
        "r2_mean": float(fold_df["r2"].mean()),
        "r2_std": float(fold_df["r2"].std(ddof=0)),
        "fold_metrics_path": str(cv_dir / "fold_metrics.csv"),
    }
    save_json(aggregate, cv_dir / "aggregate_metrics.json")
    return aggregate


def compare_models(candidate_metrics: dict[str, float], production_uploaded: dict[str, Any]) -> dict[str, Any]:
    approved_metrics = load_json(PRODUCTION_METRICS_PATH)
    reference_name = "approved_project_metrics"
    reference_metrics = {
        "mae": approved_metrics.get("mae"),
        "rmse": approved_metrics.get("rmse"),
        "r2": approved_metrics.get("r2"),
    }

    if production_uploaded.get("available"):
        reference_name = "production_model_on_uploaded_test_split"
        reference_metrics = production_uploaded["metrics"]

    if not reference_metrics.get("rmse"):
        return {
            "reference": reference_name,
            "recommendation": "review_manually",
            "reason": "No valid reference RMSE was available for automatic comparison.",
        }

    rmse_improvement = (reference_metrics["rmse"] - candidate_metrics["rmse"]) / reference_metrics["rmse"]
    mae_not_worse = candidate_metrics["mae"] <= reference_metrics["mae"]
    r2_not_worse = candidate_metrics["r2"] >= reference_metrics["r2"]
    should_update = rmse_improvement >= 0.05 and mae_not_worse

    return {
        "reference": reference_name,
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "rmse_improvement_fraction": float(rmse_improvement),
        "rmse_improvement_percent": float(rmse_improvement * 100),
        "mae_not_worse": bool(mae_not_worse),
        "r2_not_worse": bool(r2_not_worse),
        "recommendation": "recommend_update" if should_update else "keep_current_model",
        "decision_rule": "Recommend update only if RMSE improves by at least 5% and MAE is not worse.",
    }


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return MODEL_REGISTRY_DIR / "candidates" / f"candidate_{timestamp}"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    raw_df = pd.read_csv(args.dataset)
    validate_dataset(raw_df)
    cycle_df = aggregate_to_cycles(raw_df)
    train_raw, val_raw, test_raw, split_strategy = split_batteries(cycle_df, args.seed)
    validate_split_has_windows(train_raw, val_raw, test_raw)

    scaler = fit_scaler(train_raw)
    train_df = apply_scaler(train_raw, scaler)
    val_df = apply_scaler(val_raw, scaler)
    test_df = apply_scaler(test_raw, scaler)

    X_train, y_train, train_metadata = create_windows(train_df)
    X_val, y_val, val_metadata = create_windows(val_df)
    X_test, y_test, test_metadata = create_windows(test_df)

    model, history, y_pred = train_and_predict(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    candidate_metrics = metrics_dict(y_test, y_pred)
    production_uploaded = evaluate_production_model(X_test, y_test)
    comparison = compare_models(candidate_metrics, production_uploaded)
    cross_validation = (
        run_leave_one_battery_out_cv(
            cycle_df,
            output_dir,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        if args.cross_validate
        else {"available": False, "reason": "Cross-validation was not requested."}
    )

    predictions_df = test_metadata.copy()
    predictions_df["true_RUL"] = y_test
    predictions_df["predicted_RUL"] = y_pred
    predictions_df["prediction_error"] = predictions_df["predicted_RUL"] - predictions_df["true_RUL"]

    config = CandidateConfig(
        uploaded_dataset_path=str(args.dataset),
        output_dir=str(output_dir),
        split_strategy=split_strategy,
        feature_columns=FEATURE_COLUMNS,
        window_size=WINDOW_SIZE,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        dense_units=DENSE_UNITS,
        seed=args.seed,
    )

    dataset_summary = {
        "rows": int(len(raw_df)),
        "cycle_rows_after_aggregation": int(len(cycle_df)),
        "battery_ids": sorted(cycle_df["battery_id"].unique().tolist()),
        "split_strategy": split_strategy,
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
    }
    summary = {
        "candidate_metrics": candidate_metrics,
        "production_on_uploaded_test": production_uploaded,
        "comparison": comparison,
        "cross_validation": cross_validation,
        "dataset_summary": dataset_summary,
        "config": asdict(config),
    }

    save_json(candidate_metrics, output_dir / "metrics.json")
    save_json(summary, output_dir / "comparison_summary.json")
    save_json(asdict(config), output_dir / "config.json")
    save_json({key: list(value) for key, value in scaler.items()}, output_dir / "scaler_params.json")

    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)
    train_metadata.to_csv(output_dir / "train_window_metadata.csv", index=False)
    val_metadata.to_csv(output_dir / "validation_window_metadata.csv", index=False)
    test_metadata.to_csv(output_dir / "test_window_metadata.csv", index=False)

    model.save(output_dir / "candidate_glu_two_blocks_pooling_model.keras")
    with (output_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    plot_losses(history, output_dir / "loss_curve.png")
    plot_true_vs_pred(y_test, y_pred, output_dir / "true_vs_pred.png")
    plot_prediction_error(test_metadata, y_test, y_pred, output_dir / "prediction_error_over_cycles.png")

    print(json.dumps(summary, indent=2))
    print(f"Saved candidate model results to: {output_dir}")


if __name__ == "__main__":
    main()
