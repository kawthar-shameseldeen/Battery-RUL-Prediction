from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


DATA_DIR = Path("data/processed")


def load_dataset(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    return df


def get_battery_series(df: pd.DataFrame, battery_id: str, target_col: str = "SOH") -> pd.Series:
    battery_df = df[df["battery_id"] == battery_id].copy()
    battery_df = battery_df.sort_values("cycle").reset_index(drop=True)
    series = battery_df[target_col]
    series.index = battery_df["cycle"]
    return series


def adf_test(series: pd.Series, title: str = "Series") -> None:
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


def plot_series(series: pd.Series, title: str = "Time Series") -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values, color="navy")
    plt.title(title)
    plt.xlabel("Cycle")
    plt.ylabel(series.name if series.name else "Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(series: pd.Series, lags: int = 30) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.tight_layout()
    plt.show()


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}
