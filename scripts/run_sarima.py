from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utils_sarima import load_dataset


DATA_FILE = "train_dataset.csv"
BATTERY_ID = "B5"
TARGET_COL = "SOH"

# Start simple. You can change these later after testing.
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 1, 10)


OUTPUT_DIR = Path("results/sarima")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def main():
    print("Step 1: loading dataset")
    df = load_dataset(DATA_FILE)

    print("Step 2: filtering one battery")
    battery_df = df[df["battery_id"] == BATTERY_ID].copy()
    battery_df = battery_df.sort_values("cycle").reset_index(drop=True)

    # Keep cycle values for plotting only
    cycle_values = battery_df["cycle"].reset_index(drop=True)

    # Use integer index for modeling to avoid statsmodels index warnings
    series = battery_df[TARGET_COL].reset_index(drop=True)

    split_idx = int(len(series) * 0.8)

    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]

    train_cycles = cycle_values.iloc[:split_idx]
    test_cycles = cycle_values.iloc[split_idx:]

    print(f"Battery: {BATTERY_ID}")
    print("Train length:", len(train_series))
    print("Test length:", len(test_series))

    print("Step 3: plotting series")
    plt.figure(figsize=(10, 4))
    plt.plot(cycle_values, series, color="navy")
    plt.title(f"{BATTERY_ID} {TARGET_COL} Series")
    plt.xlabel("Cycle")
    plt.ylabel(TARGET_COL)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "series_plot.png", dpi=150)
    plt.show()

    print("Step 4: ADF test")
    adf_test(train_series, f"{BATTERY_ID} {TARGET_COL}")

    print("Step 5: ACF/PACF")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train_series.dropna(), lags=30, ax=axes[0])
    plot_pacf(train_series.dropna(), lags=30, ax=axes[1], method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "acf_pacf.png", dpi=150)
    plt.show()

    print("Step 6: fitting SARIMA model")
    model = SARIMAX(
        train_series,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(disp=False)
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

    results_df = pd.DataFrame({
        "cycle": test_cycles.values,
        "actual_soh": test_series.values,
        "forecast_soh": forecast.values,
    })
    results_df.to_csv(OUTPUT_DIR / "sarima_forecast.csv", index=False)

    with open(OUTPUT_DIR / "sarima_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"BATTERY_ID={BATTERY_ID}\n")
        f.write(f"ORDER={ORDER}\n")
        f.write(f"SEASONAL_ORDER={SEASONAL_ORDER}\n")
        f.write(f"RMSE={rmse}\n")
        f.write(f"MAE={mae}\n")
        f.write(f"MEAN_ACTUAL={mean_actual}\n")
        f.write(f"ACCURACY_LIKE_PERCENT={accuracy_like}\n")

    print("Step 8: plotting forecast")
    plt.figure(figsize=(10, 4))
    plt.plot(train_cycles, train_series, label="Train", color="blue")
    plt.plot(test_cycles, test_series, label="Actual Test", color="green")
    plt.plot(test_cycles, forecast, label="Forecast", color="orange")
    plt.title(f"SARIMA Forecast: {BATTERY_ID} train/test split")
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sarima_forecast.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
