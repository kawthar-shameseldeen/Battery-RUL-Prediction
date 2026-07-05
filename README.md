# Battery RUL Prediction

This project predicts battery Remaining Useful Life (RUL) from battery-cycle
measurements. The current GLU experiments use cycle-level windows and compare
several Gated Linear Unit (GLU) architectures under the same data setup.

## Data Setup Used for GLU

The selected processed data is based on these files:

- `data/processed/train_dataset.csv`
- `data/processed/validation_dataset.csv`
- `data/processed/test_dataset.csv`
- `data/processed/windows_w10.npz`

The current split is:

| Split | Batteries |
|---|---|
| Train | B5, B7 |
| Validation | B18 |
| Test | B6 |

The model input features are:

```text
chI, chV, chT, disI, disV, BCt, SOH
```

The following columns are not used as model input:

```text
cycle, battery_id, RUL, disT
```

`cycle` is used only to sort cycles and build windows. `battery_id` is used only
to separate batteries and prevent windows from crossing battery boundaries. `RUL`
is the prediction target. `disT` was removed because it is constant.

## Sequence Definition

The original processed rows contain multiple rows per cycle, so the data is first
aggregated by:

```text
battery_id + cycle
```

Mean aggregation is used. After aggregation, each battery-cycle becomes one row.

The GLU models use sliding windows:

- `window_size = 10`
- one time step = one battery cycle
- one input sample shape = `(10, 7)`
- one batch shape = `(32, 10, 7)`
- target `y` = `RUL` at the last cycle of the window

Example:

```text
X = cycles 1-10 features
y = RUL at cycle 10
```

## Training Settings

All GLU experiments used the same main training settings:

| Setting | Value |
|---|---:|
| Loss | MSE |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 100 |
| Early stopping patience | 10 |
| Random seed | 42 |

The goal was to keep the data, split, target, metrics, and training settings the
same, then change only the architecture.

## What GLU Does

GLU means Gated Linear Unit. It learns two parts:

```text
GLU(X) = A * sigmoid(B)
```

`A` is the learned feature branch. `B` is the gate branch. The sigmoid converts
`B` into values between 0 and 1, then those gate values decide how much of `A`
should pass through.

In this project, GLU is useful because battery degradation patterns are nonlinear.
The gate can help the model suppress less useful signals and keep stronger
degradation-related patterns.

## GLU Experiments

We started with a small GLU model and gradually improved it. The purpose was not
to jump directly to a large architecture, but to understand what kind of change
actually helps.

### 1. Small GLU

The first model used one small GLU block:

```text
Input -> GLU -> Dropout -> Last timestep -> Dense -> RUL
```

It trained successfully, which proved that the data pipeline and GLU block worked.
However, the model was too small. It mostly predicted middle-range RUL values,
underpredicting high RUL and overpredicting low RUL.

Result:

| MAE | RMSE | R2 |
|---:|---:|---:|
| 35.89 | 43.92 | 0.084 |

Interpretation: this was a valid baseline, but it underfit badly.

### 2. Medium GLU

The next model increased the hidden dimension from 16 to 32 and used a slightly
stronger dense head:

```text
Input -> GLU -> Dropout -> Last timestep -> Dense -> Dense -> RUL
```

This improved performance a lot. The model learned a wider range of predictions
and followed the RUL trend better.

Result:

| MAE | RMSE | R2 |
|---:|---:|---:|
| 29.33 | 33.38 | 0.471 |

Interpretation: increasing capacity helped, and the medium model became the first
strong GLU baseline.

### 3. Large GLU

The large model increased the hidden dimension to 64 and added more capacity:

```text
Input -> GLU -> Dropout -> Dense over time -> Dropout -> Last timestep -> Dense -> Dense -> RUL
```

This model was much bigger, but it did not improve the test results.

Result:

| MAE | RMSE | R2 |
|---:|---:|---:|
| 29.85 | 34.03 | 0.450 |

Interpretation: simply making the model larger did not help. The model had more
parameters, but generalization was slightly worse than the medium model.

### 4. Medium GLU + Pooling

The next improvement changed how the sequence was summarized. Instead of using
only the last time step, the model used:

```text
last timestep + average pooling
```

Architecture:

```text
Input -> GLU -> Dropout -> Last timestep + Average pooling -> Dense -> Dense -> RUL
```

This helped because the model could use both the final cycle condition and the
overall 10-cycle pattern.

Result:

| MAE | RMSE | R2 |
|---:|---:|---:|
| 27.98 | 32.02 | 0.513 |

Interpretation: pooling was a strong improvement. It showed that sequence summary
was a weakness in the previous models.

### 5. Two GLU Blocks + Pooling

The next model added a second GLU block while keeping the medium hidden size:

```text
Input -> GLU Block 1 -> Dropout -> GLU Block 2 -> Dropout -> Last timestep + Average pooling -> Dense -> Dense -> RUL
```

This gave the model a deeper gated representation without making it as wide as the
large model.

Result:

| MAE | RMSE | R2 |
|---:|---:|---:|
| 26.17 | 30.50 | 0.559 |

Interpretation: this became the best single-split GLU model. It improved all
three metrics: lowest MAE, lowest RMSE, and highest R2.

### 6. Two GLU Blocks + Residual + Pooling

We also tried adding a residual connection:

```text
Input -> GLU Block 1 -> GLU Block 2 -> Add GLU Block 1 output -> Pooling -> Dense -> RUL
```

The goal was to preserve useful features from the first GLU block while adding
the deeper transformation from the second block.

Result:

| MAE | RMSE | R2 |
|---:|---:|---:|
| 27.44 | 31.88 | 0.518 |

Interpretation: the residual version trained correctly, but it did not beat the
plain two-block pooling model. The skip connection did not help this dataset.

## Single-Split GLU Comparison

| Model | Trainable Params | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| Small GLU | 401 | 35.89 | 43.92 | 0.084 |
| Medium GLU | 1,185 | 29.33 | 33.38 | 0.471 |
| Large GLU | 7,809 | 29.85 | 34.03 | 0.450 |
| Medium GLU + Pooling | 1,697 | 27.98 | 32.02 | 0.513 |
| Two GLU Blocks + Pooling | 3,809 | 26.17 | 30.50 | 0.559 |
| Two GLU Blocks + Residual + Pooling | 3,809 | 27.44 | 31.88 | 0.518 |

The best single-split model is:

```text
Two GLU Blocks + Pooling
```

It has the lowest error and the highest R2:

- `MAE = 26.17`
- `RMSE = 30.50`
- `R2 = 0.559`

This model is selected as the best GLU architecture so far.

## Cross-Validation

After selecting the best GLU model, we ran leave-one-battery-out cross-validation.
This means each battery becomes the unseen test battery once.

The selected cross-validation model was:

```text
Two GLU Blocks + Pooling
```

For each fold:

- one battery was used for testing
- one battery was used for validation
- the remaining two batteries were used for training
- scaling was fit only on the training batteries
- windows were created inside each battery only

### Fold Results

| Fold | Train | Validation | Test | MAE | RMSE | R2 |
|---:|---|---|---|---:|---:|---:|
| 1 | B7, B18 | B6 | B5 | 17.30 | 23.11 | 0.747 |
| 2 | B5, B7 | B18 | B6 | 26.90 | 31.44 | 0.531 |
| 3 | B6, B18 | B5 | B7 | 15.86 | 21.05 | 0.790 |
| 4 | B5, B6 | B7 | B18 | 20.60 | 23.43 | 0.565 |

### Aggregate Cross-Validation Results

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| MAE | 20.17 | 4.91 |
| RMSE | 24.76 | 4.58 |
| R2 | 0.658 | 0.129 |

Final CV summary:

```text
MAE  = 20.17 +/- 4.91
RMSE = 24.76 +/- 4.58
R2   = 0.658 +/- 0.129
```

Interpretation:

- The model generalizes reasonably well across unseen batteries.
- The hardest test battery was B6.
- The best folds were B5 and B7 as test batteries.
- The standard deviation shows that performance changes depending on which
  battery is held out, which is expected with only four batteries.
- The cross-validation average is stronger than the single B6 test result,
  meaning the model is not only learning one lucky split.

## Main Conclusion

The best GLU model is:

```text
Two GLU Blocks + Pooling
```

The development path showed an important pattern:

- small model underfit
- medium model improved substantially
- large model did not help
- pooling improved sequence usage
- a second GLU block improved feature learning
- residual connection did not improve generalization

The best improvement was not simply increasing size. The best improvement was
using the sequence better and adding one extra gated transformation.

## XAI for the Approved GLU Model

After supervisor approval, the next step is explainable AI (XAI) for the
selected model:

```text
Two GLU Blocks + Pooling
```

The XAI script uses two complementary methods:

- permutation importance: measures how much RMSE worsens when one feature is
  shuffled
- integrated gradients: estimates how strongly each feature and each step in the
  10-cycle window contributes to the prediction

Run:

```powershell
python scripts/explain_glu_two_blocks_pooling.py
```

The script writes outputs to:

```text
results/xai_glu_two_blocks_pooling
```

Important XAI outputs:

- `permutation_importance.csv`
- `integrated_gradients_feature_importance.csv`
- `integrated_gradients_temporal_importance.csv`
- `integrated_gradients_heatmap.csv`
- `xai_summary.json`
- `permutation_importance.png`
- `integrated_gradients_feature_importance.png`
- `integrated_gradients_temporal_importance.png`
- `integrated_gradients_heatmap.png`

## Result Files

Important result folders:

- `results/glu_small`
- `results/glu_medium`
- `results/glu_large`
- `results/glu_medium_pooling`
- `results/glu_two_blocks_pooling`
- `results/glu_two_blocks_residual_pooling`
- `results/glu_cv_leave_one_battery_out`
- `results/glu_summary_plots`

Important summary plots:

- `results/glu_summary_plots/model_mae_comparison.png`
- `results/glu_summary_plots/model_rmse_comparison.png`
- `results/glu_summary_plots/model_r2_comparison.png`
- `results/glu_summary_plots/model_metrics_grouped.png`
- `results/glu_summary_plots/model_size_vs_rmse.png`
- `results/glu_summary_plots/cv_fold_mae_rmse.png`
- `results/glu_summary_plots/cv_fold_r2.png`
- `results/glu_summary_plots/cv_aggregate_metrics.png`

Important summary tables:

- `results/glu_summary_plots/glu_model_comparison_metrics.csv`
- `results/glu_summary_plots/cv_fold_metrics.csv`
- `results/glu_cv_leave_one_battery_out/fold_metrics.csv`
- `results/glu_cv_leave_one_battery_out/aggregate_metrics.json`

## Scripts

Main GLU scripts:

- `scripts/train_glu_small.py`
- `scripts/train_glu_medium.py`
- `scripts/train_glu_large.py`
- `scripts/train_glu_medium_pooling.py`
- `scripts/train_glu_two_blocks_pooling.py`
- `scripts/train_glu_two_blocks_residual_pooling.py`
- `scripts/cross_validate_glu_two_blocks_pooling.py`
- `scripts/plot_glu_results_summary.py`
- `scripts/explain_glu_two_blocks_pooling.py`

To rerun the selected best single-split GLU model:

```powershell
python scripts/train_glu_two_blocks_pooling.py
```

To rerun leave-one-battery-out cross-validation:

```powershell
python scripts/cross_validate_glu_two_blocks_pooling.py
```

To regenerate the summary plots:

```powershell
python scripts/plot_glu_results_summary.py
```

To generate XAI outputs for the approved GLU model:

```powershell
python scripts/explain_glu_two_blocks_pooling.py
```

## Repository Structure

- `data/processed/` → processed dataset, train/test datasets, and split file
- `notebook/` → preprocessing notebook
- `figures/` → saved plots and visualizations
- `results/` → model outputs and evaluation results
