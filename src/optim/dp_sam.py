from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F

from src.dp.accounting import create_accountant, get_sample_rate
from src.utils.metrics import accuracy

try:
    from opacus.grad_sample import GradSampleModule
except Exception:  # pragma: no cover
    from opacus import GradSampleModule


# DPSAT (Differentially Private Sharpness-Aware Training, ICML 2023)-style DP-SAM.
# We compute DP gradients twice per batch (perturb + update) and account for both.


def wrap_model_for_grad_sample(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, GradSampleModule):
        return model
    return GradSampleModule(model)


def clear_grad_samples(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if hasattr(p, "grad_sample"):
            p.grad_sample = None


def compute_dp_grads(
    model: torch.nn.Module,
    max_grad_norm: float,
    noise_multiplier: float,
) -> Tuple[List[torch.Tensor], float]:
    grad_samples = []
    for p in model.parameters():
        if hasattr(p, "grad_sample") and p.grad_sample is not None:
            grad_samples.append(p.grad_sample)

    if not grad_samples:
        return [], 0.0

    batch_size = grad_samples[0].shape[0]
    device = grad_samples[0].device
    per_sample_norms = torch.zeros(batch_size, device=device)

    for p in model.parameters():
        if not hasattr(p, "grad_sample") or p.grad_sample is None:
            continue
        gs = p.grad_sample
        per_sample_norms += gs.view(batch_size, -1).pow(2).sum(dim=1)

    per_sample_norms = per_sample_norms.sqrt()
    clip_factors = (per_sample_norms / max_grad_norm).clamp(min=1.0)

    dp_grads: List[torch.Tensor] = []
    for p in model.parameters():
        if not hasattr(p, "grad_sample") or p.grad_sample is None:
            dp_grads.append(None)
            continue
        gs = p.grad_sample
        view = [batch_size] + [1] * (gs.dim() - 1)
        clipped = gs / clip_factors.view(*view)
        grad = clipped.sum(dim=0)
        if noise_multiplier > 0:
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * max_grad_norm,
                size=grad.shape,
                device=grad.device,
            )
            grad = grad + noise
        grad = grad / float(batch_size)
        dp_grads.append(grad)

    norms = [g.norm(p=2) for g in dp_grads if g is not None]
    if not norms:
        return dp_grads, 0.0
    grad_norm = torch.sqrt(torch.sum(torch.stack(norms)))
    return dp_grads, float(grad_norm.item())


def sam_perturb_(model: torch.nn.Module, dp_grads: List[torch.Tensor], rho: float) -> List[torch.Tensor]:
    norms = [g.norm(p=2) for g in dp_grads if g is not None]
    if not norms:
        return [None for _ in model.parameters()]
    grad_norm = torch.sqrt(torch.sum(torch.stack(norms)))
    scale = rho / (grad_norm + 1e-12)
    perturbations: List[torch.Tensor] = []
    idx = 0
    for p in model.parameters():
        grad = dp_grads[idx] if idx < len(dp_grads) else None
        idx += 1
        if grad is None:
            perturbations.append(None)
            continue
        e_w = grad * scale
        p.data.add_(e_w)
        perturbations.append(e_w)
    return perturbations


def sam_restore_(model: torch.nn.Module, perturbations: List[torch.Tensor]) -> None:
    idx = 0
    for p in model.parameters():
        e_w = perturbations[idx] if idx < len(perturbations) else None
        idx += 1
        if e_w is None:
            continue
        p.data.sub_(e_w)


def apply_dp_grads(model: torch.nn.Module, dp_grads: List[torch.Tensor]) -> None:
    idx = 0
    for p in model.parameters():
        grad = dp_grads[idx] if idx < len(dp_grads) else None
        idx += 1
        if grad is None:
            continue
        p.grad = grad.detach()


def build_dpsat_components(
    model: torch.nn.Module,
    train_loader,
    cfg: dict,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, object, object, Callable]:
    model = wrap_model_for_grad_sample(model)

    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 0.1)),
        momentum=float(opt_cfg.get("momentum", 0.0)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )

    dp_cfg = cfg.get("dp", {})
    max_grad_norm = float(dp_cfg.get("max_grad_norm", 1.0))
    noise_multiplier = float(dp_cfg.get("noise_multiplier", 1.0))
    rho = float(cfg.get("sam", {}).get("rho", 0.05))

    accountant = create_accountant()
    batch_size = int(cfg.get("batch_size", 128))
    sample_rate = get_sample_rate(batch_size, len(train_loader.dataset))

    def step_fn(batch, device: torch.device) -> Tuple[float, float, int]:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        clear_grad_samples(model)

        logits = model(inputs)
        losses = F.cross_entropy(logits, targets, reduction="none")
        loss = losses.sum()
        loss.backward()

        dp_grads, _ = compute_dp_grads(model, max_grad_norm, noise_multiplier)
        perturbations = sam_perturb_(model, dp_grads, rho)
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        optimizer.zero_grad(set_to_none=True)
        clear_grad_samples(model)

        logits_perturbed = model(inputs)
        losses_perturbed = F.cross_entropy(logits_perturbed, targets, reduction="none")
        loss_perturbed = losses_perturbed.sum()
        loss_perturbed.backward()

        dp_grads_2, _ = compute_dp_grads(model, max_grad_norm, noise_multiplier)
        sam_restore_(model, perturbations)
        apply_dp_grads(model, dp_grads_2)
        optimizer.step()
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        optimizer.zero_grad(set_to_none=True)
        clear_grad_samples(model)

        batch_size = targets.size(0)
        batch_loss = float(losses.mean().item())
        batch_acc = accuracy(logits, targets)
        return batch_loss, batch_acc, batch_size

    return model, optimizer, train_loader, accountant, step_fn
