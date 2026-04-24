# SARIMA Plan

## Objective

Build and evaluate a SARIMA model to forecast battery State of Health (`SOH`) using battery `B5`.

## Dataset Used

- Main processed dataset: `data/processed/processed_data.csv`
- Training file used for the SARIMA experiment: `data/processed/train_dataset.csv`
- Battery selected for modeling: `B5`
- Target column: `SOH`
- Time index / ordering column: `cycle`

## What We Did From Start

1. Removed the previous ARIMA/SARIMA experiment files and kept the processed dataset.
2. Created the SARIMA workflow files:
   - `notebook/sarima_workflow.ipynb`
   - `scripts/utils_sarima.py`
   - `scripts/run_sarima.py`
3. Explored the battery data and confirmed that `SOH` would be the forecasting target.
4. First tried a cross-battery setup (`B5` train, `B7` test), then corrected it.
5. Changed the experiment to the proper SARIMA setup:
   - use one battery only
   - split the same battery by time
6. Selected battery `B5` and split it into 80% train and 20% test.
7. Checked stationarity using the ADF test.
8. Inspected ACF and PACF plots.
9. Chose the final SARIMA configuration:
   - `ORDER = (1, 1, 1)`
   - `SEASONAL_ORDER = (1, 0, 1, 10)`
10. Trained the SARIMA model on the `B5` training segment.
11. Forecasted the `B5` test segment.
12. Evaluated the forecast using RMSE and MAE.
13. Saved plots and output files in `results/sarima/`.

## Final Experiment Setup

- Model family: `SARIMA`
- Final model: `SARIMA(1,1,1)x(1,0,1,10)`
- Data file used in the final run: `train_dataset.csv`
- Battery used: `B5`
- Train/test method: time-based split on the same battery

## Train/Test Split

- Total observations for `B5`: `220`
- Training observations: `176`
- Testing observations: `44`

Split logic:
- first 80% of `B5` -> training set
- last 20% of `B5` -> testing set

## Stationarity Check

ADF test on `B5` `SOH`:

- ADF Statistic: `0.263118`
- p-value: `0.975588`

Critical values:
- 1%: `-3.470866`
- 5%: `-2.879330`
- 10%: `-2.576255`

Conclusion:
- The series is non-stationary.
- Because the p-value is greater than `0.05`, the null hypothesis was not rejected.
- This is why differencing was included in the model with `d = 1`.

## ACF and PACF Interpretation

- The ACF decayed slowly across many lags.
- The PACF showed a strong spike at the first lag.
- These patterns supported a differenced time-series model with small AR/MA terms.
- A seasonal structure with period `m = 10` was then tested in the final SARIMA model.

## Final Parameters

```python
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 1, 10)
```

Meaning:
- `p = 1`: one non-seasonal autoregressive term
- `d = 1`: first differencing
- `q = 1`: one non-seasonal moving average term
- `P = 1`: one seasonal autoregressive term
- `D = 0`: no seasonal differencing
- `Q = 1`: one seasonal moving average term
- `m = 10`: seasonal period of 10

## Final Console Output Values

Run command:

```bash
python scripts/run_sarima.py
```

Output summary:

- Battery: `B5`
- Train length: `176`
- Test length: `44`

ADF output:
- ADF Statistic: `0.263118`
- p-value: `0.975588`
- Conclusion: `likely non-stationary`

SARIMA summary:
- Model: `SARIMAX(1, 1, 1)x(1, 0, 1, 10)`
- Number of observations: `176`
- Log Likelihood: `619.277`
- AIC: `-1228.554`
- BIC: `-1213.085`
- HQIC: `-1222.274`

Coefficients:
- `ar.L1 = 1.0011`
- `ma.L1 = -0.9969`
- `ar.S.L10 = 0.1693`
- `ma.S.L10 = -0.2219`
- `sigma2 = 2.866e-05`

Coefficient p-values:
- `ar.L1`: `0.000`
- `ma.L1`: `0.018`
- `ar.S.L10`: `0.784`
- `ma.S.L10`: `0.717`
- `sigma2`: `0.025`

Residual diagnostics:
- Ljung-Box (L1) Q: `44.42`
- Prob(Q): `0.00`
- Jarque-Bera (JB): `1.21`
- Prob(JB): `0.55`
- Heteroskedasticity (H): `2.03`
- Prob(H) (two-sided): `0.01`
- Skew: `0.20`
- Kurtosis: `3.13`

Warnings seen during fitting:
- `ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals`
- Covariance matrix calculated using the outer product of gradients

## Forecast Performance

Forecast metrics:

- RMSE: `0.007516009629061611`
- MAE: `0.006151408311536685`
- accuracy_like_percent: `97.02579182204074%`


Interpretation:
- Both error values are very low.
- The forecast line was very close to the actual test values.
- The model performed well on the `B5` time-based test split.

## Graphs Produced

The following graphs were created and saved:

- `results/sarima/series_plot.png`
- `results/sarima/acf_pacf.png`
- `results/sarima/sarima_forecast.png`

### 1. B5 SOH Series

File:
- `results/sarima/series_plot.png`

What it showed:
- `SOH` decreases steadily as `cycle` increases.
- The battery degradation trend is clear.
- This visual trend supports the ADF result that the series is non-stationary.

### 2. ACF and PACF

File:
- `results/sarima/acf_pacf.png`

What it showed:
- ACF slowly decayed over the lags.
- PACF had a strong first lag.
- These plots supported the use of differencing and small AR/MA orders.

### 3. Forecast Graph

File:
- `results/sarima/sarima_forecast.png`

What it showed:
- Blue line: training data
- Green line: actual test data
- Orange line: SARIMA forecast
- The forecast followed the actual test values closely.

## Files Created and Used

Notebook:
- `notebook/sarima_workflow.ipynb`

Scripts:
- `scripts/utils_sarima.py`
- `scripts/run_sarima.py`

Saved outputs:
- `results/sarima/series_plot.png`
- `results/sarima/acf_pacf.png`
- `results/sarima/sarima_forecast.png`
- `results/sarima/sarima_forecast.csv`
- `results/sarima/sarima_metrics.txt`

## What Each File Did

### `notebook/sarima_workflow.ipynb`

- Loaded and explored the battery dataset.
- Filtered battery `B5`.
- Plotted the `SOH` series.
- Ran the ADF test.
- Plotted ACF and PACF.
- Tried candidate model logic and helped decide the final parameters.

### `scripts/utils_sarima.py`

- Contained helper code to load datasets.
- Kept the main execution script cleaner.

### `scripts/run_sarima.py`

- Loaded the dataset.
- Filtered battery `B5`.
- Split the series into train and test by time.
- Plotted the series.
- Ran the ADF test.
- Plotted ACF/PACF.
- Fit the SARIMA model.
- Forecasted the test segment.
- Calculated RMSE and MAE.
- Saved the forecast and plots.

## Final Conclusion

The final model selected for this task was:

```python
SARIMA(1,1,1)x(1,0,1,10)
```

This model was trained on the first 80% of battery `B5` and tested on the final 20% of the same battery. The ADF test confirmed that the original `SOH` series was non-stationary, so differencing was included with `d = 1`. The model produced strong forecasting results with:

- RMSE = `0.007516009629061611`
- MAE = `0.006151408311536685`

Overall, the SARIMA model gave an accurate forecast of the battery `SOH` trend for this experiment.
