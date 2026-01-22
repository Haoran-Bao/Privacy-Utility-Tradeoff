from typing import Dict

import torch

from .yeom import per_sample_losses, yeom_attack


def evaluate_mia(model, member_loader, nonmember_loader, device: torch.device) -> Dict[str, float]:
    member_losses = per_sample_losses(model, member_loader, device)
    nonmember_losses = per_sample_losses(model, nonmember_loader, device)
    return yeom_attack(member_losses, nonmember_losses)
