from typing import Dict

import torch


def build_sgd(model, cfg: Dict) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 0.1))
    momentum = float(opt_cfg.get("momentum", 0.0))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
