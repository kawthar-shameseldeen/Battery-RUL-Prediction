import itertools
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils_sarima import evaluate_forecast, get_battery_series, load_dataset


warnings.filterwarnings("ignore")

TRAIN_FILE = "train_dataset.csv"
TEST_FILE = "test_dataset.csv"
TRAIN_BATTERY = "B5"
TEST_BATTERY = "B7"
TARGET_COL = "SOH"

p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]
m_values = [5, 10, 20]


def main():
    train_df = load_dataset(TRAIN_FILE)
    test_df = load_dataset(TEST_FILE)

    train_series = get_battery_series(train_df, TRAIN_BATTERY, TARGET_COL)
    test_series = get_battery_series(test_df, TEST_BATTERY, TARGET_COL)

    rows = []

    for order in itertools.product(p_values, d_values, q_values):
        for seasonal_base in itertools.product(P_values, D_values, Q_values):
            for m in m_values:
                seasonal_order = seasonal_base + (m,)

                try:
                    model = SARIMAX(
                        train_series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False)
                    forecast = fitted.forecast(steps=len(test_series))
                    forecast.index = test_series.index

                    metrics = evaluate_forecast(test_series, forecast)

                    rows.append({
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "aic": fitted.aic,
                        "bic": fitted.bic,
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                    })

                    print("done:", order, seasonal_order)

                except Exception as e:
                    print("failed:", order, seasonal_order, str(e))

    results = pd.DataFrame(rows).sort_values("rmse")
    print(results.head(10).to_string(index=False))
    results.to_csv("results/sarima/sarima_search_results.csv", index=False)


if __name__ == "__main__":
    main()
