import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.aggregate import load_runs
from src.analysis.plotting import plot_metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from runs")
    parser.add_argument("--runs_dir", default="outputs/runs", help="Runs directory")
    parser.add_argument("--out_dir", default="outputs/figures", help="Output figures directory")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No runs found in {args.runs_dir}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    plot_metric(
        runs,
        metric="test_acc",
        output_path=os.path.join(args.out_dir, "test_acc.png"),
        title="Test Accuracy vs Epoch",
        ylabel="test_acc",
    )
    plot_metric(
        runs,
        metric="mia_auc",
        output_path=os.path.join(args.out_dir, "mia_auc.png"),
        title="MIA AUC vs Epoch",
        ylabel="mia_auc",
    )
    plot_metric(
        runs,
        metric="epsilon",
        output_path=os.path.join(args.out_dir, "epsilon.png"),
        title="Epsilon vs Epoch",
        ylabel="epsilon",
    )


if __name__ == "__main__":
    main()
