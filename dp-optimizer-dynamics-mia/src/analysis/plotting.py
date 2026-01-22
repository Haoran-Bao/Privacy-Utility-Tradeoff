import os
from typing import Dict, List

import matplotlib.pyplot as plt


def _label_for_record(rec: Dict) -> str:
    method = rec.get("method", "")
    eps_or_noise = rec.get("epsilon_target_or_noise_multiplier", "")
    seed = rec.get("seed", "")
    return f"{method}-x{eps_or_noise}-s{seed}"


def plot_metric(runs: Dict[str, List[Dict]], metric: str, output_path: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    for run_id, records in runs.items():
        xs = []
        ys = []
        for rec in records:
            value = rec.get(metric)
            if value is None:
                continue
            xs.append(rec.get("epoch"))
            ys.append(value)
        if not xs:
            continue
        label = _label_for_record(records[0])
        plt.plot(xs, ys, label=label)

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
