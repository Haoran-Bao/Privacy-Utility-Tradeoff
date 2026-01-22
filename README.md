# DP Optimizer Dynamics + MIA (CIFAR-10)

Minimal PyTorch repo to compare DP-SGD vs DP-SAM on CIFAR-10 with Yeom loss-threshold membership inference evaluated over the training trajectory.

## Install

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Quick start

```bash
python scripts/run_one.py --config configs/dp_sgd.yaml eps=5 seed=0
python scripts/run_one.py --config configs/dp_sam.yaml seed=0
python scripts/run_sweep.py --config configs/sweep.yaml
python scripts/make_plots.py --runs_dir outputs/runs
```

## Notes
- DP-SGD uses Opacus `PrivacyEngine` and reports epsilon at each epoch.
- DP-SAM uses fixed noise by default and reports the resulting epsilon from an Opacus accountant. Target-epsilon mode is stubbed (TODO).
- MIA is Yeom loss-threshold attack using train members vs test non-members (default).

## Outputs
- Per-run metrics: `outputs/runs/<run_id>/metrics.jsonl`
- Checkpoints: `outputs/runs/<run_id>/checkpoints/epoch_XXX.pt`
- Plots: `outputs/figures/*.png`
