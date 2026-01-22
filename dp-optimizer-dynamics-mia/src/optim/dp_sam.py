from typing import List, Tuple

import torch

try:
    from opacus.grad_sample import GradSampleModule
except Exception:  # pragma: no cover
    from opacus import GradSampleModule


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
