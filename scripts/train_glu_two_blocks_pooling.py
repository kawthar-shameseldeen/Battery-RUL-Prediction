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
WINDOWS_PATH = WORKSPACE / "data" / "processed" / "windows_w10.npz"
TEST_METADATA_PATH = WORKSPACE / "data" / "processed" / "test_window_metadata_w10.csv"
RESULTS_DIR = WORKSPACE / "results" / "glu_two_blocks_pooling"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001
HIDDEN_DIM = 32
DROPOUT = 0.15
DENSE_UNITS = 16


@dataclass
class TrainingConfig:
    windows_path: str
    test_metadata_path: str
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    hidden_dim: int
    dropout: float
    dense_units: int
    glu_blocks: int
    sequence_summary: str
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


def load_windows(path: Path):
    data = np.load(path, allow_pickle=True)
    return {
        "X_train": data["X_train"].astype(np.float32),
        "y_train": data["y_train"].astype(np.float32),
        "X_val": data["X_val"].astype(np.float32),
        "y_val": data["y_val"].astype(np.float32),
        "X_test": data["X_test"].astype(np.float32),
        "y_test": data["y_test"].astype(np.float32),
        "feature_columns": data["feature_columns"].tolist(),
        "window_size": int(data["window_size"]),
    }


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


def plot_losses(history, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("GLU Two Blocks + Pooling: Training vs Validation Loss")
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
    plt.title("GLU Two Blocks + Pooling: True RUL vs Predicted RUL")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_prediction_error(metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    errors = y_pred - y_true
    plt.figure(figsize=(9, 5))
    for battery_id, group in metadata.assign(error=errors).groupby("battery_id"):
        plt.plot(group["end_cycle"], group["error"], marker="o", linewidth=1.5, markersize=3, label=battery_id)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("End Cycle of Window")
    plt.ylabel("Prediction Error (Predicted - True)")
    plt.title("GLU Two Blocks + Pooling: Prediction Error over Cycles")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_json(data: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    windows = load_windows(WINDOWS_PATH)
    test_metadata = pd.read_csv(TEST_METADATA_PATH)

    X_train = windows["X_train"]
    y_train = windows["y_train"]
    X_val = windows["X_val"]
    y_val = windows["y_val"]
    X_test = windows["X_test"]
    y_test = windows["y_test"]

    if len(test_metadata) != len(y_test):
        raise ValueError(
            f"Test metadata length {len(test_metadata)} does not match y_test length {len(y_test)}."
        )

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
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "test_loss_mse": float(test_eval[0]),
        "test_metric_mae": float(test_eval[1]),
        "feature_columns": windows["feature_columns"],
        "window_size": windows["window_size"],
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
    }

    config = TrainingConfig(
        windows_path=str(WINDOWS_PATH),
        test_metadata_path=str(TEST_METADATA_PATH),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        dense_units=DENSE_UNITS,
        glu_blocks=2,
        sequence_summary="last_timestep_plus_global_average_pooling",
        seed=SEED,
    )

    predictions_df = test_metadata.copy()
    predictions_df["true_RUL"] = y_test
    predictions_df["predicted_RUL"] = y_pred
    predictions_df["prediction_error"] = predictions_df["predicted_RUL"] - predictions_df["true_RUL"]

    save_json(metrics, RESULTS_DIR / "metrics.json")
    save_json(asdict(config), RESULTS_DIR / "config.json")
    with (RESULTS_DIR / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))
    pd.DataFrame(history.history).to_csv(RESULTS_DIR / "history.csv", index=False)
    predictions_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    model.save(RESULTS_DIR / "glu_two_blocks_pooling_model.keras")

    plot_losses(history, RESULTS_DIR / "loss_curve.png")
    plot_true_vs_pred(y_test, y_pred, RESULTS_DIR / "true_vs_pred.png")
    plot_prediction_error(test_metadata, y_test, y_pred, RESULTS_DIR / "prediction_error_over_cycles.png")

    print(json.dumps(metrics, indent=2))
    print(f"Saved results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
