import os
from typing import Dict, Optional

import torch

from src.data.cifar10 import build_datasets, build_loaders
from src.data.split import build_mia_loaders
from src.dp.accounting import default_delta, find_noise_multiplier, get_accountant_epsilon, get_sample_rate
from src.dp.opacus_engine import get_epsilon, make_private, make_private_with_epsilon
from src.mia.eval import evaluate_mia
from src.models.factory import create_model
from src.optim.dp_sam import build_dpsat_components
from src.optim.dp_sgd import build_sgd
from src.optim.sam import SAM
from src.train.loop import evaluate, train_epoch_dp_sgd, train_epoch_sam_non_dp, train_epoch_with_step_fn
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import log_jsonl
from src.utils.seed import resolve_device, set_seed


def _get_delta(cfg: Dict, train_size: int) -> float:
    dp_cfg = cfg.get("dp", {})
    delta = dp_cfg.get("delta")
    if delta is None:
        return default_delta(train_size)
    return float(delta)


def _format_run_id(cfg: Dict) -> str:
    method = cfg.get("method", "dp_sgd")
    seed = cfg.get("seed", 0)
    dp_cfg = cfg.get("dp", {})
    privacy_mode = dp_cfg.get("privacy_mode", "fixed_noise")
    if privacy_mode == "target_epsilon":
        tag = f"eps{dp_cfg.get('target_epsilon', 'na')}"
    else:
        tag = f"nm{dp_cfg.get('noise_multiplier', 'na')}"
    return f"{method}_{tag}_seed{seed}"


def run_experiment(cfg: Dict, run_id: Optional[str] = None) -> str:
    project_root = cfg.get("project_root", ".")
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg.get("device", "auto"))

    train_set, test_set = build_datasets(cfg)
    train_loader, test_loader = build_loaders(train_set, test_set, cfg)
    member_loader, nonmember_loader = build_mia_loaders(train_set, test_set, cfg)

    model = create_model(cfg).to(device)

    method = cfg.get("method", "dp_sgd")
    dp_cfg = cfg.get("dp", {})
    max_grad_norm = float(dp_cfg.get("max_grad_norm", 1.0))
    noise_multiplier = float(dp_cfg.get("noise_multiplier", 1.0))
    privacy_mode = dp_cfg.get("privacy_mode", "fixed_noise")

    optimizer = build_sgd(model, cfg)
    privacy_engine = None
    accountant = None
    sample_rate = get_sample_rate(int(cfg.get("batch_size", 128)), len(train_set))
    allow_non_dp = bool(cfg.get("dp_sam", {}).get("allow_non_dp", False))

    use_non_dp_sam = False
    dpsat_step_fn = None

    if method == "dp_sgd":
        if privacy_mode == "target_epsilon":
            model, optimizer, train_loader, privacy_engine = make_private_with_epsilon(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                target_epsilon=float(dp_cfg.get("target_epsilon", 5.0)),
                target_delta=_get_delta(cfg, len(train_set)),
                epochs=int(cfg.get("epochs", 1)),
                max_grad_norm=max_grad_norm,
                poisson_sampling=False,
            )
            nm = getattr(privacy_engine, "noise_multiplier", None)
            if nm is None:
                nm = getattr(optimizer, "noise_multiplier", None)
            if nm is not None:
                noise_multiplier = float(nm)
        else:
            model, optimizer, train_loader, privacy_engine = make_private(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                poisson_sampling=False,
            )
    elif method in {"dp_sam", "dpsat"}:
        if privacy_mode == "target_epsilon":
            steps = int(cfg.get("epochs", 1)) * len(train_loader) * 2
            noise_multiplier = find_noise_multiplier(
                target_epsilon=float(dp_cfg.get("target_epsilon", 5.0)),
                target_delta=_get_delta(cfg, len(train_set)),
                sample_rate=sample_rate,
                steps=steps,
            )
            dp_cfg["noise_multiplier"] = float(noise_multiplier)
            cfg["dp"] = dp_cfg
        try:
            model, optimizer, train_loader, accountant, dpsat_step_fn = build_dpsat_components(
                model=model,
                train_loader=train_loader,
                cfg=cfg,
            )
        except Exception as exc:
            if not allow_non_dp:
                raise RuntimeError(
                    "Opacus grad-sample support unavailable. Set dp_sam.allow_non_dp=true to run non-DP SAM."
                ) from exc
            optimizer = SAM(
                model.parameters(),
                torch.optim.SGD,
                rho=float(cfg.get("sam", {}).get("rho", 0.05)),
                **cfg.get("optimizer", {}),
            )
            use_non_dp_sam = True
    else:
        raise ValueError(f"Unknown method: {method}")

    runs_dir = cfg.get("output", {}).get("runs_dir", "outputs/runs")
    run_id = run_id or _format_run_id(cfg)
    run_dir = os.path.join(project_root, runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    cfg_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg, f)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    mia_every = int(cfg.get("mia", {}).get("eval_every", 1))
    epochs = int(cfg.get("epochs", 1))
    delta = _get_delta(cfg, len(train_set))

    for epoch in range(1, epochs + 1):
        if method == "dp_sgd":
            train_metrics = train_epoch_dp_sgd(model, train_loader, optimizer, device)
            epsilon = get_epsilon(privacy_engine, delta) if privacy_engine else float("nan")
        else:
            if use_non_dp_sam:
                train_metrics = train_epoch_sam_non_dp(model, train_loader, optimizer, device)
                epsilon = float("nan")
            else:
                train_metrics = train_epoch_with_step_fn(model, train_loader, dpsat_step_fn, device)
                epsilon = get_accountant_epsilon(accountant, delta) if accountant else float("nan")

        test_metrics = evaluate(model, test_loader, device)

        mia_metrics = {"mia_auc": None, "mia_advantage": None}
        if mia_every > 0 and epoch % mia_every == 0:
            mia_metrics = evaluate_mia(model, member_loader, nonmember_loader, device)

        eps_or_noise = (
            float(dp_cfg.get("target_epsilon"))
            if privacy_mode == "target_epsilon"
            else noise_multiplier
        )

        record = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_metrics.get("train_loss"),
            "train_acc": train_metrics.get("train_acc"),
            "test_loss": test_metrics.get("test_loss"),
            "test_acc": test_metrics.get("test_acc"),
            "epsilon": epsilon,
            "delta": delta,
            "mia_auc": mia_metrics.get("mia_auc"),
            "mia_advantage": mia_metrics.get("mia_advantage"),
            "method": method,
            "seed": int(cfg.get("seed", 0)),
            "epsilon_target_or_noise_multiplier": eps_or_noise,
            "max_grad_norm": max_grad_norm,
        }
        if method in {"dp_sam", "dpsat"}:
            record["rho"] = float(cfg.get("sam", {}).get("rho", 0.05))

        log_jsonl(metrics_path, record)

        if epoch % int(cfg.get("save_every", 1)) == 0:
            ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "config": cfg,
                },
                ckpt_path,
            )

    return run_dir
