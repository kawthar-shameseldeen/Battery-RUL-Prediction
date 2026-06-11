from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_REQUIRED_COLUMNS = [
    "cycle",
    "chI",
    "chV",
    "chT",
    "disI",
    "disV",
    "disT",
    "BCt",
    "SOH",
    "RUL",
]

DROP_MODEL_FEATURE_COLUMNS = {"disT"}

ALTERNATE_COLUMN_MAP = {
    "cycle": "cycle",
    "ambient_temperature": "ambient_temperature",
    "capacity": "capacity",
    "voltage_measured": "voltage_measured",
    "current_measured": "current_measured",
    "temperature_measured": "temperature_measured",
    "current_load": "current_load",
    "voltage_load": "voltage_load",
    "time": "time",
    "RUL": "RUL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a folder of battery CSV files into the raw dataset expected by the "
            "project and rebuild the processed train/test files."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/source_csv",
        help="Folder containing the new CSV files, for example B0005_discharge.csv.",
    )
    parser.add_argument(
        "--raw-output",
        default="data/raw/Battery_dataset.csv",
        help="Path for the merged raw dataset.",
    )
    parser.add_argument(
        "--processed-output",
        default="data/processed/processed_data.csv",
        help="Path for the scaled processed dataset.",
    )
    parser.add_argument(
        "--train-output",
        default="data/processed/train_dataset.csv",
        help="Path for the processed training split.",
    )
    parser.add_argument(
        "--test-output",
        default="data/processed/test_dataset.csv",
        help="Path for the processed testing split.",
    )
    parser.add_argument(
        "--split-file",
        default="data/processed/split.json",
        help="JSON file containing train_batteries and test_batteries.",
    )
    parser.add_argument(
        "--keep-leading-zeros",
        action="store_true",
        help="Keep battery IDs like B0005 instead of normalizing them to B5.",
    )
    return parser.parse_args()


def normalize_battery_id(raw_id: str, keep_leading_zeros: bool) -> str:
    raw_id = str(raw_id).strip()
    match = re.search(r"([Bb])0*(\d+)", raw_id)
    if not match:
        return raw_id
    prefix, number = match.groups()
    if keep_leading_zeros:
        return f"{prefix.upper()}{number.zfill(len(raw_id) - 1)}"
    return f"{prefix.upper()}{int(number)}"


def battery_id_from_filename(path: Path, keep_leading_zeros: bool) -> str:
    match = re.search(r"([Bb])0*(\d+)", path.stem)
    if not match:
        raise ValueError(
            f"Could not infer battery ID from filename '{path.name}'. "
            "Add a 'battery_id' column or rename the file to include a battery ID."
        )
    prefix, number = match.groups()
    if keep_leading_zeros:
        return f"{prefix.upper()}{number.zfill(len(match.group(0)) - 1)}"
    return f"{prefix.upper()}{int(number)}"


def validate_columns(df: pd.DataFrame, source_name: str) -> None:
    missing = [column for column in DEFAULT_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {', '.join(missing)}"
        )


def has_alternate_schema(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in ALTERNATE_COLUMN_MAP)


def transform_alternate_schema(df: pd.DataFrame) -> pd.DataFrame:
    transformed = pd.DataFrame()
    transformed["cycle"] = pd.to_numeric(df["cycle"], errors="raise")
    transformed["chI"] = pd.to_numeric(df["current_measured"], errors="raise")
    transformed["chV"] = pd.to_numeric(df["voltage_measured"], errors="raise")
    transformed["chT"] = pd.to_numeric(df["temperature_measured"], errors="raise")
    transformed["disI"] = pd.to_numeric(df["current_load"], errors="raise")
    transformed["disV"] = pd.to_numeric(df["voltage_load"], errors="raise")
    transformed["disT"] = pd.to_numeric(df["ambient_temperature"], errors="raise")
    transformed["BCt"] = pd.to_numeric(df["capacity"], errors="raise")

    initial_capacity = transformed.groupby(df["battery_id"])["BCt"].transform("max")
    transformed["SOH"] = transformed["BCt"] / initial_capacity
    transformed["battery_id"] = df["battery_id"]
    transformed["RUL"] = pd.to_numeric(df["RUL"], errors="raise")
    return transformed


def load_and_merge_csvs(input_dir: Path, keep_leading_zeros: bool) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files were found in '{input_dir}'.")

    merged_frames: list[pd.DataFrame] = []
    reference_columns: list[str] | None = None

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "battery_id" not in df.columns:
            df["battery_id"] = battery_id_from_filename(csv_path, keep_leading_zeros)
        df["battery_id"] = df["battery_id"].map(
            lambda value: normalize_battery_id(value, keep_leading_zeros)
        )

        if has_alternate_schema(df):
            df = transform_alternate_schema(df)
        else:
            validate_columns(df, csv_path.name)

        ordered_columns = DEFAULT_REQUIRED_COLUMNS[:-1] + ["battery_id", "RUL"]
        current_columns = ordered_columns

        if reference_columns is None:
            reference_columns = current_columns
        elif current_columns != reference_columns:
            raise ValueError(
                f"{csv_path.name} does not match the expected column order/shape.\n"
                f"Expected: {reference_columns}\n"
                f"Found:    {current_columns}"
            )

        merged_frames.append(df[reference_columns])

    merged = pd.concat(merged_frames, ignore_index=True)
    return merged


def fit_min_max_params(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    feature_columns = [col for col in df.columns if col not in {"battery_id", "RUL"}]
    params: dict[str, tuple[float, float]] = {}

    for column in feature_columns:
        numeric = pd.to_numeric(df[column], errors="raise")
        params[column] = (numeric.min(), numeric.max())

    return params


def drop_model_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in DROP_MODEL_FEATURE_COLUMNS if col in df.columns])


def min_max_scale(
    df: pd.DataFrame, params: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    processed = df.copy()

    for column, (column_min, column_max) in params.items():
        numeric = pd.to_numeric(processed[column], errors="raise")
        if column_max == column_min:
            processed[column] = 0.0
        else:
            processed[column] = (numeric - column_min) / (column_max - column_min)

    processed["RUL"] = pd.to_numeric(processed["RUL"], errors="raise")
    return processed


def load_split(split_file: Path) -> tuple[list[str], list[str]]:
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file '{split_file}' was not found. Create it before rebuilding train/test files."
        )

    payload = json.loads(split_file.read_text(encoding="utf-8"))
    return payload.get("train_batteries", []), payload.get("test_batteries", [])


def validate_split(battery_ids: set[str], train_ids: list[str], test_ids: list[str]) -> None:
    train_set = set(train_ids)
    test_set = set(test_ids)

    overlap = sorted(train_set & test_set)
    if overlap:
        raise ValueError(
            "The same battery IDs appear in both train_batteries and test_batteries: "
            + ", ".join(overlap)
        )

    unknown = sorted((train_set | test_set) - battery_ids)
    if unknown:
        raise ValueError(
            "split.json contains battery IDs that are not present in the new dataset: "
            + ", ".join(unknown)
        )

    unassigned = sorted(battery_ids - (train_set | test_set))
    if unassigned:
        raise ValueError(
            "Some batteries in the new dataset are missing from split.json: "
            + ", ".join(unassigned)
        )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    raw_output = Path(args.raw_output)
    processed_output = Path(args.processed_output)
    train_output = Path(args.train_output)
    test_output = Path(args.test_output)
    split_file = Path(args.split_file)

    merged = load_and_merge_csvs(input_dir, args.keep_leading_zeros)
    train_ids, test_ids = load_split(split_file)
    battery_ids = set(merged["battery_id"].unique())
    validate_split(battery_ids, train_ids, test_ids)

    train_raw = drop_model_feature_columns(merged[merged["battery_id"].isin(train_ids)])
    test_raw = drop_model_feature_columns(merged[merged["battery_id"].isin(test_ids)])
    merged_model = drop_model_feature_columns(merged)

    scaling_params = fit_min_max_params(train_raw)
    processed = min_max_scale(merged_model, scaling_params)
    train_dataset = min_max_scale(train_raw, scaling_params)
    test_dataset = min_max_scale(test_raw, scaling_params)

    for output_path in [raw_output, processed_output, train_output, test_output]:
        ensure_parent(output_path)

    merged.to_csv(raw_output, index=False)
    processed.to_csv(processed_output, index=False)
    train_dataset.to_csv(train_output, index=False)
    test_dataset.to_csv(test_output, index=False)

    print(f"Merged {len(merged)} rows from {input_dir}.")
    print(f"Battery IDs: {sorted(battery_ids)}")
    print(f"Raw dataset written to: {raw_output}")
    print(f"Processed dataset written to: {processed_output}")
    print(f"Train dataset rows: {len(train_dataset)} -> {train_output}")
    print(f"Test dataset rows: {len(test_dataset)} -> {test_output}")


if __name__ == "__main__":
    main()
