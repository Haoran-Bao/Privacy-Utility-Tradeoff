from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.optim.dp_sam import (
    apply_dp_grads,
    clear_grad_samples,
    compute_dp_grads,
    sam_perturb_,
    sam_restore_,
)
from src.optim.sam import SAM
from src.utils.metrics import accuracy


def train_epoch_dp_sgd(
    model,
    train_loader,
    optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        clear_grad_samples(model)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        total_acc += accuracy(logits, targets) * batch_size

    return {
        "train_loss": total_loss / max(1, total),
        "train_acc": total_acc / max(1, total),
    }


def train_epoch_dp_sam(
    model,
    train_loader,
    optimizer,
    device: torch.device,
    max_grad_norm: float,
    noise_multiplier: float,
    rho: float,
    accountant,
    sample_rate: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    # DP-SAM uses two DP gradient computations per step; this worsens privacy vs DP-SGD.
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        losses = F.cross_entropy(logits, targets, reduction="none")
        loss = losses.mean()
        loss.backward()

        dp_grads, _ = compute_dp_grads(model, max_grad_norm, noise_multiplier)
        perturbations = sam_perturb_(model, dp_grads, rho)
        if accountant is not None:
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        optimizer.zero_grad(set_to_none=True)
        clear_grad_samples(model)

        logits_perturbed = model(inputs)
        losses_perturbed = F.cross_entropy(logits_perturbed, targets, reduction="none")
        loss_perturbed = losses_perturbed.mean()
        loss_perturbed.backward()

        dp_grads_2, _ = compute_dp_grads(model, max_grad_norm, noise_multiplier)
        sam_restore_(model, perturbations)
        apply_dp_grads(model, dp_grads_2)
        optimizer.step()
        if accountant is not None:
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        optimizer.zero_grad(set_to_none=True)
        clear_grad_samples(model)

        batch_size = targets.size(0)
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        total_acc += accuracy(logits, targets) * batch_size

    return {
        "train_loss": total_loss / max(1, total),
        "train_acc": total_acc / max(1, total),
    }


def train_epoch_sam_non_dp(
    model,
    train_loader,
    optimizer: SAM,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        loss.backward()
        optimizer.first_step(zero_grad=True)

        logits_perturbed = model(inputs)
        loss_perturbed = F.cross_entropy(logits_perturbed, targets, reduction="mean")
        loss_perturbed.backward()
        optimizer.second_step(zero_grad=True)

        batch_size = targets.size(0)
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        total_acc += accuracy(logits, targets) * batch_size

    return {
        "train_loss": total_loss / max(1, total),
        "train_acc": total_acc / max(1, total),
    }


def evaluate(model, data_loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets, reduction="mean")

            batch_size = targets.size(0)
            total += batch_size
            total_loss += float(loss.item()) * batch_size
            total_acc += accuracy(logits, targets) * batch_size

    return {
        "test_loss": total_loss / max(1, total),
        "test_acc": total_acc / max(1, total),
    }
