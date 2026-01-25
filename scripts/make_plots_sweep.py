import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.aggregate import load_runs


def _norm_method(method: str) -> str:
    if method in {"dp_sam", "dpsat"}:
        return "dp_sam"
    return method


def _eps_key(value) -> float:
    return round(float(value), 6)


def _collect_metric(records: List[Dict], metric: str) -> Dict[str, Dict[float, Dict[int, List[float]]]]:
    data: Dict[str, Dict[float, Dict[int, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        method = _norm_method(rec.get("method", ""))
        eps_val = rec.get("epsilon_target_or_noise_multiplier")
        if eps_val is None:
            continue
        epoch = rec.get("epoch")
        value = rec.get(metric)
        if epoch is None or value is None:
            continue
        data[method][_eps_key(eps_val)][int(epoch)].append(float(value))
    return data


def _mean_series(epoch_values: Dict[int, List[float]]) -> Tuple[List[int], List[float]]:
    epochs = sorted(epoch_values.keys())
    means = []
    for ep in epochs:
        vals = epoch_values[ep]
        means.append(sum(vals) / max(1, len(vals)))
    return epochs, means


def plot_grid(
    records: List[Dict],
    epsilons: List[float],
    metric: str,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    metric_data = _collect_metric(records, metric)
    methods = ["dp_sgd", "dp_sam"]
    colors = {"dp_sgd": "tab:blue", "dp_sam": "tab:orange"}

    fig, axes = plt.subplots(1, len(epsilons), figsize=(5 * len(epsilons), 4), squeeze=False)

    for idx, eps in enumerate(epsilons):
        ax = axes[0][idx]
        eps_key = _eps_key(eps)
        for method in methods:
            if method not in metric_data or eps_key not in metric_data[method]:
                continue
            epochs, means = _mean_series(metric_data[method][eps_key])
            ax.plot(epochs, means, label=method, color=colors.get(method))
        ax.set_title(f"epsilon={eps}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_eps_list(raw: str) -> List[float]:
    if not raw:
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep grids for dp_sgd vs dp_sam")
    parser.add_argument("--runs_dir", default="outputs/runs", help="Runs directory")
    parser.add_argument("--out_dir", default="outputs/figures_sweep", help="Output figures directory")
    parser.add_argument("--epsilons", default="", help="Comma-separated list of target epsilons")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No runs found in {args.runs_dir}")
        return

    records = []
    for run_id, recs in runs.items():
        records.extend(recs)

    eps_list = _parse_eps_list(args.epsilons)
    if not eps_list:
        eps_list = sorted({_eps_key(r.get("epsilon_target_or_noise_multiplier")) for r in records if r.get("epsilon_target_or_noise_multiplier") is not None})

    if not eps_list:
        print("No epsilon values found in records.")
        return

    plot_grid(
        records,
        eps_list,
        metric="test_acc",
        title="Test Accuracy vs Epoch (DP-SGD vs DP-SAM)",
        ylabel="test_acc",
        out_path=os.path.join(args.out_dir, "test_acc_grid.png"),
    )
    plot_grid(
        records,
        eps_list,
        metric="mia_auc",
        title="MIA AUC vs Epoch (DP-SGD vs DP-SAM)",
        ylabel="mia_auc",
        out_path=os.path.join(args.out_dir, "mia_auc_grid.png"),
    )
    plot_grid(
        records,
        eps_list,
        metric="epsilon",
        title="Epsilon vs Epoch (DP-SGD vs DP-SAM)",
        ylabel="epsilon",
        out_path=os.path.join(args.out_dir, "epsilon_grid.png"),
    )


if __name__ == "__main__":
    main()
