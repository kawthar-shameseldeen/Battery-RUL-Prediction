# Battery-RUL-Prediction

## Replacing the dataset safely

This project currently expects one merged raw file at [data/raw/Battery_dataset.csv](C:\Users\hp\Desktop\Battery-RUL-Prediction\data\raw\Battery_dataset.csv), even if your source data comes as many CSV files.

### Where to put the new files

1. Copy all new battery CSV files into [data/source_csv](C:\Users\hp\Desktop\Battery-RUL-Prediction\data\source_csv).
2. Keep one CSV per battery, for example `B0005_discharge.csv`, `B0006_discharge.csv`, and so on.
3. If the CSV files do not already contain a `battery_id` column, the script will infer it from the filename.

### What will change

Running the replacement script rebuilds these files:

- [data/raw/Battery_dataset.csv](C:\Users\hp\Desktop\Battery-RUL-Prediction\data\raw\Battery_dataset.csv): one merged raw dataset from all CSV files.
- [data/processed/processed_data.csv](C:\Users\hp\Desktop\Battery-RUL-Prediction\data\processed\processed_data.csv): min-max scaled feature dataset.
- [data/processed/train_dataset.csv](C:\Users\hp\Desktop\Battery-RUL-Prediction\data\processed\train_dataset.csv): rows selected by `train_batteries`.
- `data/processed/validation_dataset.csv`: rows selected by `validation_batteries`.
- [data/processed/test_dataset.csv](C:\Users\hp\Desktop\Battery-RUL-Prediction\data\processed\test_dataset.csv): rows selected by `test_batteries`.

The split file is still [data/processed/split.json](C:\Users\hp\Desktop\Battery-RUL-Prediction\data\processed\split.json). It defines `train_batteries`, `validation_batteries`, and `test_batteries`. If the new dataset contains different battery IDs, update that JSON before rebuilding.

Feature scaling is fit on the training batteries only, then applied to the train, validation, test, and combined processed files. This avoids leaking unseen-battery statistics into the training pipeline.

The constant `disT` feature is kept in the raw merged dataset but removed from processed model-ready datasets, because it has only one value and cannot help the GRU/GLU learn degradation behavior.

The `cycle` column is kept unscaled as an ordering column for sequence/window creation, but it should not be used as a model input feature.

### How to rebuild

Run:

```powershell
python scripts/replace_dataset.py
```

If your new files use IDs like `B0005` and you want to keep the leading zeros, run:

```powershell
python scripts/replace_dataset.py --keep-leading-zeros
```

### Important validation

The script stops instead of silently creating bad splits when:

- required columns are missing
- CSV files do not share the same structure
- a battery appears in more than one split
- a battery in the new dataset is missing from `split.json`
- `split.json` references a battery that is not in the new dataset
