import itertools
import argparse
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils_sarima import evaluate_forecast, get_battery_series, load_dataset


warnings.filterwarnings("ignore")

DEFAULT_TRAIN_FILE = "train_dataset.csv"
DEFAULT_TEST_FILE = "test_dataset.csv"
DEFAULT_TRAIN_BATTERY = "B5"
DEFAULT_TEST_BATTERY = "B7"
DEFAULT_TARGET_COL = "SOH"

p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]
m_values = [5, 10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search SARIMA settings.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--train-battery", default=DEFAULT_TRAIN_BATTERY)
    parser.add_argument("--test-battery", default=DEFAULT_TEST_BATTERY)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument(
        "--output-file",
        default="results/sarima/sarima_search_results.csv",
        help="CSV file for sorted SARIMA search results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = load_dataset(args.train_file)
    test_df = load_dataset(args.test_file)

    train_series = get_battery_series(train_df, args.train_battery, args.target_col)
    test_series = get_battery_series(test_df, args.test_battery, args.target_col)

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
    results.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
