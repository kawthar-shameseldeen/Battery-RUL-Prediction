import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


DATA_DIR = Path("data/processed")


def parse_tuple(text: str, expected_length: int, option_name: str) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != expected_length:
        raise ValueError(
            f"{option_name} must contain exactly {expected_length} comma-separated integers."
        )
    return tuple(int(part) for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ARIMA experiment on one battery.")
    parser.add_argument(
        "--data-file",
        default="train_dataset.csv",
        help="CSV file name inside data/processed.",
    )
    parser.add_argument(
        "--battery-id",
        default="B5",
        help="Battery ID to filter, for example B5, B6, B7, or B18.",
    )
    parser.add_argument(
        "--target-col",
        default="SOH",
        help="Target column to forecast. Default is SOH.",
    )
    parser.add_argument(
        "--order",
        default="1,1,1",
        help="ARIMA order as p,d,q.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train ratio for the selected battery time series. Must be between 0 and 1.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/arima",
        help="Directory where plots and forecast files will be saved.",
    )
    return parser.parse_args()


def load_dataset(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    return pd.read_csv(path)


def adf_test(series: pd.Series, title: str) -> None:
    result = adfuller(series.dropna())
    print(f"ADF Test: {title}")
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.6f}")

    if result[1] < 0.05:
        print("Conclusion: likely stationary")
    else:
        print("Conclusion: likely non-stationary")


def main() -> None:
    args = parse_args()
    order = parse_tuple(args.order, 3, "--order")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < args.split_ratio < 1:
        raise ValueError("--split-ratio must be between 0 and 1.")

    print("Step 1: loading dataset")
    df = load_dataset(args.data_file)

    print("Step 2: filtering one battery")
    battery_df = df[df["battery_id"] == args.battery_id].copy()
    if battery_df.empty:
        available_batteries = sorted(df["battery_id"].astype(str).unique().tolist())
        raise ValueError(
            f"Battery '{args.battery_id}' was not found in {args.data_file}. "
            f"Available batteries: {available_batteries}"
        )
    battery_df = battery_df.sort_values("cycle").reset_index(drop=True)

    cycle_values = battery_df["cycle"].reset_index(drop=True)
    series = battery_df[args.target_col].reset_index(drop=True)

    split_idx = int(len(series) * args.split_ratio)
    if split_idx <= 0 or split_idx >= len(series):
        raise ValueError(
            f"Split ratio {args.split_ratio} produced an invalid split for {len(series)} rows."
        )

    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]
    train_cycles = cycle_values.iloc[:split_idx]
    test_cycles = cycle_values.iloc[split_idx:]

    print(f"Dataset file: {args.data_file}")
    print(f"Battery: {args.battery_id}")
    print("Train length:", len(train_series))
    print("Test length:", len(test_series))

    print("Step 3: plotting series")
    plt.figure(figsize=(10, 4))
    plt.plot(cycle_values, series, color="navy")
    plt.title(f"{args.battery_id} {args.target_col} Series")
    plt.xlabel("Cycle")
    plt.ylabel(args.target_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "series_plot.png", dpi=150)
    plt.show()

    print("Step 4: ADF test")
    adf_test(train_series, f"{args.battery_id} {args.target_col}")

    print("Step 5: ACF/PACF")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train_series.dropna(), lags=30, ax=axes[0])
    plot_pacf(train_series.dropna(), lags=30, ax=axes[1], method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.tight_layout()
    plt.savefig(output_dir / "acf_pacf.png", dpi=150)
    plt.show()

    print("Step 6: fitting ARIMA model")
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    print(fitted_model.summary())

    print("Step 7: forecasting")
    forecast = fitted_model.forecast(steps=len(test_series))
    forecast = pd.Series(forecast).reset_index(drop=True)

    rmse = mean_squared_error(test_series, forecast, squared=False)
    mae = mean_absolute_error(test_series, forecast)
    mean_actual = test_series.mean()
    accuracy_like = (1 - mae / mean_actual) * 100

    print("Forecast metrics:")
    print({"rmse": rmse, "mae": mae, "accuracy_like_percent": accuracy_like})

    results_df = pd.DataFrame(
        {
            "cycle": test_cycles.values,
            "actual_soh": test_series.values,
            "forecast_soh": forecast.values,
        }
    )
    results_df.to_csv(output_dir / "arima_forecast.csv", index=False)

    with open(output_dir / "arima_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"DATA_FILE={args.data_file}\n")
        f.write(f"BATTERY_ID={args.battery_id}\n")
        f.write(f"TARGET_COL={args.target_col}\n")
        f.write(f"ORDER={order}\n")
        f.write(f"SPLIT_RATIO={args.split_ratio}\n")
        f.write(f"RMSE={rmse}\n")
        f.write(f"MAE={mae}\n")
        f.write(f"MEAN_ACTUAL={mean_actual}\n")
        f.write(f"ACCURACY_LIKE_PERCENT={accuracy_like}\n")

    print("Step 8: plotting forecast")
    plt.figure(figsize=(10, 4))
    plt.plot(train_cycles, train_series, label="Train", color="blue")
    plt.plot(test_cycles, test_series, label="Actual Test", color="green")
    plt.plot(test_cycles, forecast, label="Forecast", color="orange")
    plt.title(f"ARIMA Forecast: {args.battery_id} train/test split")
    plt.xlabel("Cycle")
    plt.ylabel(args.target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "arima_forecast.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
