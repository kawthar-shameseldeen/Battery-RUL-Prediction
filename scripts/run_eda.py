import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_FILE = "data/raw/Battery_dataset.csv"
DEFAULT_OUTPUT_DIR = "results/eda"


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

    plt.figure(figsize=(10, 5))
    for battery_id in sorted(df["battery_id"].astype(str).unique()):
        subset = df[df["battery_id"] == battery_id].sort_values("cycle")
        plt.plot(subset["cycle"], subset["RUL"], label=battery_id)
    plt.title("RUL vs Cycle for All Batteries")
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "all_batteries_rul_vs_cycle.png", dpi=150)
    plt.show()

    numeric_df = df.select_dtypes(include=["number"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.show()

    numeric_cols = [col for col in numeric_df.columns if col != "RUL"] + ["RUL"]
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df[numeric_cols])
    plt.title("Boxplots of Numeric Features")
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_boxplots.png", dpi=150)
    plt.show()

    print(f"\nEDA figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
