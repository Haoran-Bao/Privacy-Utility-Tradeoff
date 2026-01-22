from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def per_sample_losses(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            batch_losses = F.cross_entropy(logits, targets, reduction="none")
            losses.append(batch_losses.detach().cpu().numpy())
    if not losses:
        return np.array([])
    return np.concatenate(losses, axis=0)


def yeom_attack(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> Dict[str, float]:
    if member_losses.size == 0 or nonmember_losses.size == 0:
        return {"mia_auc": float("nan"), "mia_advantage": float("nan")}
    labels = np.concatenate([
        np.ones_like(member_losses, dtype=np.int32),
        np.zeros_like(nonmember_losses, dtype=np.int32),
    ])
    scores = -np.concatenate([member_losses, nonmember_losses])
    auc = roc_auc_score(labels, scores)
    advantage = 2.0 * auc - 1.0
    return {"mia_auc": float(auc), "mia_advantage": float(advantage)}
