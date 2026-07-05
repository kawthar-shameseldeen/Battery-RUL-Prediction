import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_FILE = "data/raw/Battery_dataset.csv"
DEFAULT_OUTPUT_DIR = "results/eda"
DROP_MODEL_FEATURE_COLUMNS = {"disT"}


def build_cycle_rul_df(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["battery_id", "cycle"], as_index=False)["RUL"]
        .mean()
        .sort_values(["battery_id", "cycle"])
    )


def find_overlapping_rul_trends(cycle_df: pd.DataFrame) -> list[list[str]]:
    signatures: dict[tuple[tuple[float, float], ...], list[str]] = {}

    for battery_id in sorted(cycle_df["battery_id"].astype(str).unique()):
        subset = cycle_df[cycle_df["battery_id"].astype(str) == battery_id]
        signature = tuple(
            (round(float(cycle), 8), round(float(rul), 8))
            for cycle, rul in zip(subset["cycle"], subset["RUL"])
        )
        signatures.setdefault(signature, []).append(battery_id)

    return [battery_ids for battery_ids in signatures.values() if len(battery_ids) > 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick EDA on the merged battery dataset before training."
    )
    parser.add_argument(
        "--data-file",
        default=DEFAULT_DATA_FILE,
        help="Path to the merged raw dataset CSV.",
    )
    parser.add_argument(
        "--battery-id",
        default="B5",
        help="Battery ID for the single-battery RUL vs cycle plot.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved EDA figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_file)
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Battery IDs:", sorted(df["battery_id"].astype(str).unique().tolist()))
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe(include="all"))

    battery_df = df[df["battery_id"] == args.battery_id].copy()
    if battery_df.empty:
        raise ValueError(
            f"Battery '{args.battery_id}' was not found. "
            f"Available batteries: {sorted(df['battery_id'].astype(str).unique().tolist())}"
        )
    battery_df = battery_df.sort_values("cycle")

    plt.figure(figsize=(10, 5))
    plt.plot(battery_df["cycle"], battery_df["RUL"], color="navy")
    plt.title(f"RUL vs Cycle for {args.battery_id}")
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.battery_id}_rul_vs_cycle.png", dpi=150)
    plt.show()

    cycle_rul_df = build_cycle_rul_df(df)
    overlapping_groups = find_overlapping_rul_trends(cycle_rul_df)

    if overlapping_groups:
        print("\nOverlapping RUL trends:")
        for group in overlapping_groups:
            print(f"- {', '.join(group)} have the same RUL trend")
    else:
        print("\nNo identical RUL trends found.")

    overlap_offsets = {}
    for group in overlapping_groups:
        center = (len(group) - 1) / 2
        for idx, battery_id in enumerate(group):
            overlap_offsets[battery_id] = (idx - center) * 0.8

    plt.figure(figsize=(10, 5))
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "x", "P", "*"]
    for idx, battery_id in enumerate(sorted(cycle_rul_df["battery_id"].astype(str).unique())):
        subset = cycle_rul_df[cycle_rul_df["battery_id"].astype(str) == battery_id]
        y_values = subset["RUL"] + overlap_offsets.get(battery_id, 0.0)
        plt.plot(
            subset["cycle"],
            y_values,
            label=battery_id,
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],
            markevery=max(len(subset) // 12, 1),
            linewidth=2,
            markersize=5,
        )
    plt.title("RUL vs Cycle for All Batteries")
    plt.xlabel("Cycle")
    plt.ylabel("RUL (small visual offsets for identical overlaps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "all_batteries_rul_vs_cycle.png", dpi=150)
    plt.show()

    model_df = df.drop(
        columns=[col for col in DROP_MODEL_FEATURE_COLUMNS if col in df.columns]
    )
    numeric_df = model_df.select_dtypes(include=["number"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.show()

    numeric_cols = [col for col in numeric_df.columns if col != "RUL"] + ["RUL"]
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=model_df[numeric_cols])
    plt.title("Boxplots of Numeric Features")
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_boxplots.png", dpi=150)
    plt.show()

    print(f"\nEDA figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
