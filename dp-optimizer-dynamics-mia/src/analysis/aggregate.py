import glob
import json
import os
from typing import Dict, List


def load_metrics_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_runs(runs_dir: str) -> Dict[str, List[Dict]]:
    runs: Dict[str, List[Dict]] = {}
    pattern = os.path.join(runs_dir, "*", "metrics.jsonl")
    for metrics_path in glob.glob(pattern):
        records = load_metrics_jsonl(metrics_path)
        for rec in records:
            run_id = rec.get("run_id") or os.path.basename(os.path.dirname(metrics_path))
            runs.setdefault(run_id, []).append(rec)
    for run_id in runs:
        runs[run_id] = sorted(runs[run_id], key=lambda r: r.get("epoch", 0))
    return runs
