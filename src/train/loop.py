from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

from src.optim.dp_sam import clear_grad_samples
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


def train_epoch_with_step_fn(
    model,
    train_loader,
    step_fn: Callable,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for inputs, targets in train_loader:
        batch_loss, batch_acc, batch_size = step_fn((inputs, targets), device)
        total += batch_size
        total_loss += float(batch_loss) * batch_size
        total_acc += float(batch_acc) * batch_size

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
