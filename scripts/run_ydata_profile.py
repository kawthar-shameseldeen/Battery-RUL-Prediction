import argparse
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a ydata-profiling HTML report for the battery dataset."
    )
    parser.add_argument(
        "--data-file",
        default="data/raw/Battery_dataset.csv",
        help="Path to the CSV file to profile.",
    )
    parser.add_argument(
        "--output-file",
        default="results/eda/ydata_profile_report.html",
        help="Path to the HTML report output.",
    )
    parser.add_argument(
        "--title",
        default="Battery Dataset Profile Report",
        help="Report title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_file = Path(args.data_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    profile = ProfileReport(
        df,
        title=args.title,
        explorative=True,
    )
    profile.to_file(output_file)

    print(f"Profile report created: {output_file}")


if __name__ == "__main__":
    main()
